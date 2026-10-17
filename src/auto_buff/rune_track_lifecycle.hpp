#ifndef AUTO_AIM_ENERGY_RUNE_TRACK_LIFECYCLE_HPP
#define AUTO_AIM_ENERGY_RUNE_TRACK_LIFECYCLE_HPP

#include <algorithm>
#include <cmath>

namespace auto_aim::energy
{
    enum class RuneTrackStatus { Track, TemporaryLost, Lost };

    /// @brief Time-based lifecycle for detector output.
    ///
    /// A rejected association means "this frame did not provide a usable
    /// measurement"; it does not mean the target disappeared. Keeping those
    /// concepts separate prevents one bad frame from toggling TRACK/TEMP_LOST
    /// at the camera frame rate.
    class RuneTrackLifecycle
    {
    public:
        RuneTrackLifecycle(double track_hold_s = 0.15, double lost_timeout_s = 0.8)
        {
            configure(track_hold_s, lost_timeout_s);
        }

        void configure(double track_hold_s, double lost_timeout_s)
        {
            track_hold_s_ = std::max(0.0, track_hold_s);
            lost_timeout_s_ = std::max(track_hold_s_, lost_timeout_s);
        }

        void noteAccepted(double timestamp)
        {
            last_accepted_ = timestamp;
            last_evaluated_ = timestamp;
            rejected_frames_ = 0;
        }

        void noteRejected(double timestamp)
        {
            last_evaluated_ = timestamp;
            ++rejected_frames_;
        }

        RuneTrackStatus statusAt(double timestamp) const
        {
            if (!std::isfinite(last_accepted_)) return RuneTrackStatus::Lost;
            const double age = std::max(0.0, timestamp - last_accepted_);
            if (age <= track_hold_s_) return RuneTrackStatus::Track;
            if (age <= lost_timeout_s_) return RuneTrackStatus::TemporaryLost;
            return RuneTrackStatus::Lost;
        }

        RuneTrackStatus currentStatus() const
        {
            return statusAt(last_evaluated_);
        }

        int rejectedFrames() const { return rejected_frames_; }

    private:
        double track_hold_s_ = 0.15;
        double lost_timeout_s_ = 0.8;
        double last_accepted_ = -1e300;
        double last_evaluated_ = -1e300;
        int rejected_frames_ = 0;
    };
} // namespace auto_aim::energy

#endif // AUTO_AIM_ENERGY_RUNE_TRACK_LIFECYCLE_HPP
