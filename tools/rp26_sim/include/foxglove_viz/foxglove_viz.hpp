#pragma once
//
// 影子头文件：替换 RP-26Rune 的 foxglove_viz（依赖公开仓库里缺失的
// lib/libfoxglove.a）。能量机关算法的可视化全部在 `VizTopic::*::enabled()` 为 true
// 时才会调用，这里提供同名接口的空实现，保证算法本体可以独立编译运行。
//
// 接口对照：DataPublisher / EntityPublisher / FoxgloveServer / global_foxglove_server，
// 以及 foxglove::schemas 里被算法直接构造的几个基础类型。

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "time/time.hpp"

namespace foxglove::schemas
{
struct Vector3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Vector3() = default;
    Vector3(double x_in, double y_in, double z_in) : x(x_in), y(y_in), z(z_in) {}
};

struct Quaternion
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 1.0;
};

struct Pose
{
    std::optional<Vector3> position;
    std::optional<Quaternion> orientation;

    Pose() = default;
    explicit Pose(const Vector3 &position_in) : position(position_in) {}
};

struct Color
{
    float r = 0.0F;
    float g = 0.0F;
    float b = 0.0F;
    float a = 0.0F;

    Color() = default;
    Color(float r_in, float g_in, float b_in, float a_in) : r(r_in), g(g_in), b(b_in), a(a_in) {}
};

struct SpherePrimitive
{
    std::optional<Pose> pose;
    Vector3 size;
    Color color;
};

struct ArrowPrimitive
{
    std::optional<Pose> pose;
    double shaft_length = 0.0;
    double shaft_diameter = 0.0;
    double head_length = 0.0;
    double head_diameter = 0.0;
    Color color;
};
} // namespace foxglove::schemas

namespace foxglove_viz
{
class FoxgloveServer;

/// @brief 数据 topic 发布器：算法侧只用到 publish / publish_with_time。
template <typename... Ts>
class DataPublisher
{
public:
    using Ptr = std::shared_ptr<DataPublisher<Ts...>>;

    DataPublisher(FoxgloveServer &, std::string_view, std::vector<std::string>) {}

    template <typename... Us>
    bool publish(Us &&...) { return true; }

    template <typename... Us>
    bool publish_with_time(timetool::Timestamp, Us &&...) { return true; }
};

/// @brief 场景实体发布器：算法侧只用到 publish_spheres / publish_arrows。
class EntityPublisher
{
public:
    using Ptr = std::shared_ptr<EntityPublisher>;

    template <typename Range>
    bool publish(const Range &) { return true; }

    bool publish_spheres(std::span<const foxglove::schemas::SpherePrimitive>) { return true; }
    bool publish_arrows(std::span<const foxglove::schemas::ArrowPrimitive>) { return true; }
};

/// @brief 空服务器：create_entity_publisher 返回一个可调用的空发布器。
class FoxgloveServer
{
public:
    template <typename... Ts>
    typename DataPublisher<Ts...>::Ptr create_publisher(std::string_view, const std::vector<std::string> &)
    {
        return std::make_shared<DataPublisher<Ts...>>(*this, std::string_view{}, std::vector<std::string>{});
    }

    EntityPublisher::Ptr create_entity_publisher(std::string_view, std::string_view, std::string_view)
    {
        return std::make_shared<EntityPublisher>();
    }
};

inline FoxgloveServer &global_foxglove_server()
{
    static FoxgloveServer server;
    return server;
}
} // namespace foxglove_viz
