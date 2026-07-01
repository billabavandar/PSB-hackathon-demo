//! Continuous Identity Trust — client telemetry agent (WebAssembly).
//!
//! This is the "Client-Side Telemetry" tier from the pitch. In the real product
//! it's a lightweight agent that watches how you move the mouse and type, then
//! turns a window of those raw events into the same seven behavioural-physics
//! features the Python pipeline uses. Compiling it to Wasm means the feature
//! maths lives in a compact binary blob instead of readable JS, so a client-side
//! attacker can't just open devtools and spoof "human" feature values.
//!
//! It's `no_std`: no allocator, no standard library, no panics. Everything lives
//! in a handful of fixed static buffers, which is why the release `.wasm` is only
//! a few kB. The host (JS) feeds events in, then asks for the feature vector.
//!
//! ABI (all pointers are into the exported linear memory):
//!   reset()                       clear the window buffer
//!   push_move(x, y, dt_ms)        append one pointer sample
//!   push_key(interval_ms)         append one keystroke inter-key interval
//!   move_count() / key_count()    how many events are buffered
//!   compute() -> *const f64       write 7 features to OUT, return its pointer
//!
//! OUT order matches FEATURE_COLS in features.py:
//!   [speed_mean, speed_std, accel_std, path_efficiency,
//!    angle_std, dt_std, key_interval_std]

#![no_std]

use core::f64::consts::PI;

const MAX: usize = 512;

static mut XS: [f64; MAX] = [0.0; MAX];
static mut YS: [f64; MAX] = [0.0; MAX];
static mut DTS: [f64; MAX] = [0.0; MAX];
static mut KEYS: [f64; MAX] = [0.0; MAX];
static mut NMOVE: usize = 0;
static mut NKEY: usize = 0;
static mut OUT: [f64; 7] = [0.0; 7];

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

// --- hand-rolled maths (no_std has no sqrt/atan2 on stable) --------------------

/// Square root via a bit-trick seed + a few Newton steps. Plenty accurate for
/// the value ranges we see (pixels, milliseconds).
fn sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut g = f64::from_bits((x.to_bits() + 0x3FF0_0000_0000_0000) >> 1);
    let mut i = 0;
    while i < 8 {
        g = 0.5 * (g + x / g);
        i += 1;
    }
    g
}

fn abs(x: f64) -> f64 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// arctan for the full real line, minimax polynomial on |x|<=1 with reduction.
fn atan(x: f64) -> f64 {
    if abs(x) > 1.0 {
        let s = if x < 0.0 { -1.0 } else { 1.0 };
        return s * (PI / 2.0) - atan(1.0 / x);
    }
    let x2 = x * x;
    x * (0.999866
        + x2 * (-0.330299 + x2 * (0.180141 + x2 * (-0.085133 + x2 * 0.020835))))
}

fn atan2(y: f64, x: f64) -> f64 {
    if x > 0.0 {
        atan(y / x)
    } else if x < 0.0 {
        if y >= 0.0 {
            atan(y / x) + PI
        } else {
            atan(y / x) - PI
        }
    } else if y > 0.0 {
        PI / 2.0
    } else if y < 0.0 {
        -PI / 2.0
    } else {
        0.0
    }
}

// --- ABI ----------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn reset() {
    unsafe {
        NMOVE = 0;
        NKEY = 0;
    }
}

#[no_mangle]
pub extern "C" fn push_move(x: f64, y: f64, dt_ms: f64) {
    unsafe {
        if NMOVE < MAX {
            XS[NMOVE] = x;
            YS[NMOVE] = y;
            DTS[NMOVE] = if dt_ms < 1.0 { 1.0 } else { dt_ms };
            NMOVE += 1;
        }
    }
}

#[no_mangle]
pub extern "C" fn push_key(interval_ms: f64) {
    unsafe {
        if NKEY < MAX {
            KEYS[NKEY] = interval_ms;
            NKEY += 1;
        }
    }
}

#[no_mangle]
pub extern "C" fn move_count() -> i32 {
    unsafe { NMOVE as i32 }
}

#[no_mangle]
pub extern "C" fn key_count() -> i32 {
    unsafe { NKEY as i32 }
}

/// Standard deviation of a slice via the one-pass sum / sum-of-squares identity.
fn std_of(buf: &[f64]) -> f64 {
    let n = buf.len();
    if n == 0 {
        return 0.0;
    }
    let mut s = 0.0;
    let mut ss = 0.0;
    for &v in buf {
        s += v;
        ss += v * v;
    }
    let mean = s / n as f64;
    let var = ss / n as f64 - mean * mean;
    sqrt(if var < 0.0 { 0.0 } else { var })
}

#[no_mangle]
pub extern "C" fn compute() -> *const f64 {
    unsafe {
        let n = NMOVE;
        // Not enough motion to say anything — return zeros.
        if n < 3 {
            let mut i = 0;
            while i < 7 {
                OUT[i] = 0.0;
                i += 1;
            }
            return OUT.as_ptr();
        }

        // Per-step speed, plus path length and the straight-line distance.
        // We reuse XS/YS space is risky, so accumulate on the fly.
        let mut speed_sum = 0.0;
        let mut speed_sqsum = 0.0;
        let mut accel_sqsum = 0.0;
        let mut accel_sum = 0.0;
        let mut accel_cnt = 0.0;
        let mut angle_sqsum = 0.0;
        let mut angle_sum = 0.0;
        let mut angle_cnt = 0.0;
        let mut path_len = 0.0;

        let mut prev_speed = 0.0;
        let mut prev_dx = 0.0;
        let mut prev_dy = 0.0;
        let mut have_prev_speed = false;
        let mut have_prev_seg = false;

        let mut i = 1;
        while i < n {
            let dx = XS[i] - XS[i - 1];
            let dy = YS[i] - YS[i - 1];
            let step = sqrt(dx * dx + dy * dy);
            path_len += step;
            let speed = step / DTS[i];

            speed_sum += speed;
            speed_sqsum += speed * speed;

            if have_prev_speed {
                let a = speed - prev_speed;
                accel_sum += a;
                accel_sqsum += a * a;
                accel_cnt += 1.0;
            }
            prev_speed = speed;
            have_prev_speed = true;

            // Turn angle between consecutive segments (== diff of unwrapped
            // heading), via atan2(cross, dot). Needs both segments non-trivial.
            if have_prev_seg {
                let cross = prev_dx * dy - prev_dy * dx;
                let dot = prev_dx * dx + prev_dy * dy;
                let turn = atan2(cross, dot);
                angle_sum += turn;
                angle_sqsum += turn * turn;
                angle_cnt += 1.0;
            }
            prev_dx = dx;
            prev_dy = dy;
            have_prev_seg = true;

            i += 1;
        }

        let nspeed = (n - 1) as f64;
        let speed_mean = speed_sum / nspeed;
        let speed_var = speed_sqsum / nspeed - speed_mean * speed_mean;

        let accel_std = if accel_cnt > 0.0 {
            let m = accel_sum / accel_cnt;
            let v = accel_sqsum / accel_cnt - m * m;
            sqrt(if v < 0.0 { 0.0 } else { v })
        } else {
            0.0
        };

        let angle_std = if angle_cnt > 0.0 {
            let m = angle_sum / angle_cnt;
            let v = angle_sqsum / angle_cnt - m * m;
            sqrt(if v < 0.0 { 0.0 } else { v })
        } else {
            0.0
        };

        let straight = sqrt(
            (XS[n - 1] - XS[0]) * (XS[n - 1] - XS[0])
                + (YS[n - 1] - YS[0]) * (YS[n - 1] - YS[0]),
        );

        OUT[0] = speed_mean;
        OUT[1] = sqrt(if speed_var < 0.0 { 0.0 } else { speed_var });
        OUT[2] = accel_std;
        OUT[3] = straight / (path_len + 1e-6);
        OUT[4] = angle_std;
        OUT[5] = std_of(&DTS[0..n]);
        OUT[6] = std_of(&KEYS[0..NKEY]);
        OUT.as_ptr()
    }
}
