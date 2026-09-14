# Architecture

## Data Flow

```text
IFrameSource
  -> Detector
  -> PnPSolver
  -> ArmorObservation
  -> Tracker / RotationRateEstimator
  -> TargetSelector
  -> AimSignalFilter
  -> GimbalController
  -> CommandSink
```

`TargetSelector` owns target choice and fire windows. `GimbalController` owns
gimbal trajectory generation and transport throttling. `Shooter` only consumes
the selector decision and gimbal readiness.

## Time

All runtime modules use `StandardClock::nowUs()` and local monotonic timestamps.
Remote simulator timestamps remain available through `VisionDateRecevier` for
diagnostics and stream restart detection, but are not compared with local
command-history timestamps.

## Logging

`Logger` has `DEBUG`, `INFO`, `WARNING`, and `ERROR` levels. The debug entry
enables `DEBUG`; the release entry does not. `ULTRA_VISION_LOG` optionally
writes the same log stream to a file.

## Extension Point

`energy_rune::IEnergyRuneModule` is reserved for future rune processing. It can
be injected independently without changing the auto-aim pipeline.
