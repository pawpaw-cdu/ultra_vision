# Ultra Vision

RoboMaster vision project for armor auto-aim and energy-buff detection.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

Build only one module if needed:

```bash
cmake --build build --target auto_aim -j
cmake --build build --target energy_demo -j
cmake --build build --target hsv_tuner -j
```

The top-level CMake searches for OpenCV, yaml-cpp, and the bundled Daheng
Galaxy SDK. The SDK architecture defaults to `armv8` on ARM Linux and `amd64`
elsewhere; override it with:

```bash
cmake -S . -B build -DULTRA_VISION_DAHENG_ARCH=amd64
```

`auto_aim` uses the bundled Galaxy SDK on Linux by default. On macOS it uses
the TCP simulator receiver because the Galaxy SDK is Linux-only.

## Run

```bash
./build/src/auto_aim/auto_aim
./build/src/auto_buff/energy_demo
```

`auto_aim` reads configs from `ULTRA_VISION_CONFIG_DIR` when set, or from the
first command-line argument, and otherwise uses the source `configs/` directory.

`energy_demo` accepts an optional config directory followed by a video path:

```bash
./build/src/auto_buff/energy_demo src/auto_buff/config src/auto_buff/result.mp4
```

`hsv_tuner` opens camera 0 by default, or accepts a video path as its only
argument. `Show Red` and `Show Blue` select which colors contribute to the
mask, and pressing `e` prints the tuned red/blue HSV ranges as YAML:

```bash
./build/hsv_tuner
./build/hsv_tuner test_vid_pic/buff_test.mp4
```

## Simulator

The auto-aim target can use the TCP simulator instead of the Galaxy camera.
macOS enables this mode by default; Linux and explicit builds use:

```bash
cmake -S . -B build_sim -DULTRA_VISION_AUTO_AIM_SIMULATOR=ON -DULTRA_VISION_BUILD_AUTO_BUFF=OFF
cmake --build build_sim --target auto_aim -j
```

Start the host simulator in the armor scene, then run the receiver:

```bash
cd /path/to/simulator_system/simulator
DAEDALUS_RUNE_ONLY=0 cargo run --no-default-features --features tcp-only

cd /path/to/Ultra_Vision
./build_sim/src/auto_aim/auto_aim
```

The simulator endpoint is configured in `configs/simulator.yaml`. After stable
tracking is established, `node_sim` aims at the filtered visible armor plate and
sends `GIMBAL <yaw> <pitch>` to TCP port 7667. Angles are absolute radians,
with positive yaw turning right and positive pitch turning up. The pitch limits
are configured under `tracker.gimbal` in `configs/tracker.yaml`. The receiver
enters tracking mode after stable detections and sends `FIRE` to the same port
at up to 10 Hz. Press `Esc` or `e` to exit.

The same configuration section also sets the visual-servo yaw/pitch gains,
maximum angle step per processed frame, and deadbands. The command state is
updated only after a newer image frame arrives, which prevents an asynchronous
TCP command from being mistaken for a camera pose that has already been applied.

Angle calculation lives in `src/auto_aim/control/gimbal_aimer.*`. It only
converts a target position and the current absolute gimbal angle into a new
absolute angle. TCP transmission remains in
`sim_receiver::VisionDateReceiver::sendGimbalCommand()`, so the solver can be
reused with a real gimbal control backend without simulator dependencies.
