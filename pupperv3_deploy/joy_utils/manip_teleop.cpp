#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include <algorithm>
#include <cmath>

class ManipTeleopNode : public rclcpp::Node {
public:
  ManipTeleopNode() : Node("manip_teleop_node") {
    // --- ROS Parameters ---
    this->declare_parameter<bool>("enable_two_leg", false);
    this->declare_parameter<double>("max_speed", 0.1);
    this->declare_parameter<double>("left_leg_y_offset", 0.127);
    this->declare_parameter<std::vector<double>>("x_limits", {-0.15, 0.15});
    this->declare_parameter<std::vector<double>>("y_limits", {-0.15, 0.15});
    this->declare_parameter<std::vector<double>>("z_limits", {-0.10, 0.10});

    this->get_parameter("enable_two_leg", enable_two_leg_);

    max_speed_ = (float)this->get_parameter("max_speed").as_double();
    left_leg_y_offset_ = (float)this->get_parameter("left_leg_y_offset").as_double();

    auto x_lims = this->get_parameter("x_limits").as_double_array();
    auto y_lims = this->get_parameter("y_limits").as_double_array();
    auto z_lims = this->get_parameter("z_limits").as_double_array();

    x_limit_[0] = (float)x_lims[0]; x_limit_[1] = (float)x_lims[1];
    y_limit_[0] = (float)y_lims[0]; y_limit_[1] = (float)y_lims[1];
    z_limit_[0] = (float)z_lims[0]; z_limit_[1] = (float)z_lims[1];

    // --- Subscriptions & Publishers ---
    joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
        "/joy", 10, std::bind(&ManipTeleopNode::joy_callback, this, std::placeholders::_1));

    manip_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/manip_command", 10);

    // --- State Initialization ---
    target_x_ = 0.0;
    target_z_ = 0.0;
    
    if (enable_two_leg_) {
      current_leg_id_ = 0.0;
      target_y_ = 0.0;
      RCLCPP_INFO(this->get_logger(), "Mode: Two Leg Enabled. Starting on Front-Right Leg.");
    } else {
      current_leg_id_ = 1.0; // Lock to Left Leg
      target_y_ = left_leg_y_offset_;
      RCLCPP_INFO(this->get_logger(), "Mode: Single Leg (Left). Starting at Y offset: %.3f m", target_y_);
    }

    // --- Timer ---
    // 50Hz update rate = 20ms period
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(20),
        std::bind(&ManipTeleopNode::timer_callback, this));
  }

private:
  void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg) {
    if (msg->axes.size() <= AXIS_R_STICK_V || msg->buttons.size() <= BTN_R1) {
      return; // Prevent out-of-bounds if a weird controller connects
    }

    // 1. Update velocities
    joy_vel_x_ = msg->axes[AXIS_L_STICK_V];
    joy_vel_y_ = msg->axes[AXIS_L_STICK_H];
    joy_vel_z_ = msg->axes[AXIS_R_STICK_V];

    // 2. Square Button Reset
    bool square_pressed = msg->buttons[BTN_SQUARE] == 1;
    if (square_pressed && !prev_square_) {
      target_x_ = 0.0;
      target_z_ = 0.0;
      if (enable_two_leg_) {
        target_y_ = 0.0;
        RCLCPP_INFO(this->get_logger(), "Policy reset. Targets zeroed.");
      } else {
        target_y_ = left_leg_y_offset_;
        RCLCPP_INFO(this->get_logger(), "Policy reset. Left leg returned to offset.");
      }
    }
    prev_square_ = square_pressed;

    // 3. Leg Cycling (Only in Two-Leg mode)
    bool l1_pressed = msg->buttons[BTN_L1] == 1;
    bool r1_pressed = msg->buttons[BTN_R1] == 1;

    if (enable_two_leg_) {
      if (r1_pressed && !prev_r1_) {
        current_leg_id_ = std::fmod(current_leg_id_ + 1.0, 4.0);
        target_x_ = 0.0; target_y_ = 0.0; target_z_ = 0.0;
        RCLCPP_INFO(this->get_logger(), "Switched to Leg ID: %d", (int)current_leg_id_);
      }
      if (l1_pressed && !prev_l1_) {
        current_leg_id_ = std::fmod(current_leg_id_ - 1.0 + 4.0, 4.0);
        target_x_ = 0.0; target_y_ = 0.0; target_z_ = 0.0;
        RCLCPP_INFO(this->get_logger(), "Switched to Leg ID: %d", (int)current_leg_id_);
      }
    }
    prev_l1_ = l1_pressed;
    prev_r1_ = r1_pressed;
  }

  void timer_callback() {
    // 1. Integrate position
    target_x_ += joy_vel_x_ * max_speed_ * dt_;
    target_y_ += joy_vel_y_ * max_speed_ * dt_;
    target_z_ += joy_vel_z_ * max_speed_ * dt_;

    // 2. Clamp limits
    target_x_ = std::clamp(target_x_, x_limit_[0], x_limit_[1]);
    target_y_ = std::clamp(target_y_, y_limit_[0], y_limit_[1]);
    target_z_ = std::clamp(target_z_, z_limit_[0], z_limit_[1]);

    // 3. Publish
    auto manip_msg = std_msgs::msg::Float32MultiArray();
    manip_msg.data = {target_x_, target_y_, target_z_, current_leg_id_};
    manip_pub_->publish(manip_msg);
  }

  // Configuration Constants
  const int BTN_SQUARE = 3;
  const int BTN_L1 = 4;
  const int BTN_R1 = 5;
  const int AXIS_L_STICK_H = 0;
  const int AXIS_L_STICK_V = 1;
  const int AXIS_R_STICK_V = 4;

  const float dt_ = 1.0 / 50.0;

  float max_speed_;
  float left_leg_y_offset_; // 0.127 meter
  float x_limit_[2];
  float y_limit_[2];
  float z_limit_[2];

  // State Variables
  bool enable_two_leg_;
  float current_leg_id_;
  float target_x_, target_y_, target_z_;
  float joy_vel_x_ = 0.0, joy_vel_y_ = 0.0, joy_vel_z_ = 0.0;
  bool prev_square_ = false, prev_l1_ = false, prev_r1_ = false;

  // ROS Variables
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr manip_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ManipTeleopNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}