#ifndef SIM_RECEIVER_TELEMETRY_RECEIVER_HPP
#define SIM_RECEIVER_TELEMETRY_RECEIVER_HPP

// Ground-truth channel for rune tests.
//
// The simulator broadcasts one line per rune event (hit, activation) and a few
// state lines per second on `DAEDALUS_TELEMETRY_ADDR` (default 0.0.0.0:7668).
// See the simulator's `src/telemetry.rs` for the exact format. This client
// connects once, keeps the socket open and hands the lines to the caller, so a
// test can score what actually happened on the mechanism.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sim_receiver
{
    class TelemetryReceiver
    {
    public:
        explicit TelemetryReceiver(std::string host, uint16_t port);
        ~TelemetryReceiver();

        TelemetryReceiver(const TelemetryReceiver&) = delete;
        TelemetryReceiver& operator=(const TelemetryReceiver&) = delete;

        bool connect();
        bool connected() const;

        // Lines received since the previous call.
        std::vector<std::string> poll();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace sim_receiver

#endif // SIM_RECEIVER_TELEMETRY_RECEIVER_HPP
