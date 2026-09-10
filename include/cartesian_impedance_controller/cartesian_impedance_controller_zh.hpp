#pragma once

#include <Eigen/Dense>
#include <vector>

namespace cartesian_impedance_controller
{
class CartesianImpedanceController
{
public:
  // 构造函数，初始化控制器内部状态
  CartesianImpedanceController();
  // 析构函数，使用默认实现
  ~CartesianImpedanceController() = default;

  // 直接设置期望位姿（位置与姿态），不经过滤波
  void initDesiredPose(const Eigen::Vector3d& position_d_target, const Eigen::Quaterniond& orientation_d_target);

  // 直接设置零空间关节配置目标，不经过滤波
  void initNullspaceConfig(const Eigen::VectorXd& q_d_nullspace_target);

  // 设置机器人关节数量
  void setNumberOfJoints(size_t n_joints);

  // 设置笛卡尔与零空间对角刚度，可选自动计算阻尼
  void setStiffness(const Eigen::Matrix<double, 7, 1>& stiffness, bool auto_damping = true);

  // 分别设置平移、旋转及零空间刚度，可选自动计算阻尼
  void setStiffness(double t_x, double t_y, double t_z, double r_x, double r_y, double r_z, double n,
                    bool auto_damping = true);

  // 分别设置平移与旋转刚度（不含零空间），可选自动计算阻尼
  void setStiffness(double t_x, double t_y, double t_z, double r_x, double r_y, double r_z, bool auto_damping = true);

  // 设置平移、旋转及零空间的阻尼系数
  void setDampingFactors(double d_x, double d_y, double d_z, double d_a, double d_b, double d_c, double d_n);

  // 设置末端执行器参考位姿目标，后续可通过滤波平滑过渡
  void setReferencePose(const Eigen::Vector3d& position_d, const Eigen::Quaterniond& orientation_d);

  // 设置零空间关节配置目标，后续可通过滤波平滑过渡
  void setNullspaceConfig(const Eigen::VectorXd& q_d_nullspace_target);

  // 配置零空间、刚度、位姿及力/力矩指令的滤波参数
  void setFiltering(double update_frequency, double filter_params_nullspace_config, double filter_params_stiffness,
                    double filter_params_pose, double filter_params_wrench);

  // 设置每个控制周期允许的最大力矩变化量（力矩饱和限幅）
  void setMaxTorqueDelta(double d);

  // 设置最大力矩变化量，并根据控制更新频率进行换算
  void setMaxTorqueDelta(double d, double update_frequency);

  // 在根坐标系（通常为世界系）中施加虚拟笛卡尔力/力矩
  void applyWrench(const Eigen::Matrix<double, 6, 1>& cartesian_wrench);

  // 基于已更新的内部状态计算并返回指令力矩，同时执行滤波步骤
  Eigen::VectorXd calculateCommandedTorques();

  // 更新内部状态（关节状态、末端位姿、雅可比）后计算并返回指令力矩
  Eigen::VectorXd calculateCommandedTorques(const Eigen::VectorXd& q, const Eigen::VectorXd& dq,
                                            const Eigen::Vector3d& position, Eigen::Quaterniond orientation,
                                            const Eigen::MatrixXd& jacobian);

  // 获取控制器完整状态（关节、末端、参考位姿、刚度与阻尼等）
  void getState(Eigen::VectorXd* q, Eigen::VectorXd* dq, Eigen::Vector3d* position, Eigen::Quaterniond* orientation,
                Eigen::Vector3d* position_d, Eigen::Quaterniond* orientation_d,
                Eigen::Matrix<double, 6, 6>* cartesian_stiffness, double* nullspace_stiffness,
                Eigen::VectorXd* q_d_nullspace, Eigen::Matrix<double, 6, 6>* cartesian_damping) const;

  // 获取控制器参考位姿、刚度与阻尼等状态（不含关节与当前末端位姿）
  void getState(Eigen::Vector3d* position_d, Eigen::Quaterniond* orientation_d,
                Eigen::Matrix<double, 6, 6>* cartesian_stiffness, double* nullspace_stiffness,
                Eigen::VectorXd* q_d_nullspace, Eigen::Matrix<double, 6, 6>* cartesian_damping) const;

  // 获取上一次计算得到的指令力矩
  Eigen::VectorXd getLastCommands() const;

  // 获取当前施加的笛卡尔力/力矩
  Eigen::Matrix<double, 6, 1> getAppliedWrench() const;

  // 获取当前末端位姿误差（平移与旋转）
  Eigen::Matrix<double, 6, 1> getPoseError() const;

protected:
  size_t n_joints_{ 7 };

  Eigen::Matrix<double, 6, 6> cartesian_stiffness_{ Eigen::Matrix<double, 6, 6>::Identity() };
  Eigen::Matrix<double, 6, 6> cartesian_damping_{ Eigen::Matrix<double, 6, 6>::Identity() };

  Eigen::VectorXd q_d_nullspace_;
  Eigen::VectorXd q_d_nullspace_target_;
  double nullspace_stiffness_{ 0.0 };
  double nullspace_stiffness_target_{ 0.0 };
  double nullspace_damping_{ 0.0 };
  double nullspace_damping_target_{ 0.0 };

  Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_{
    Eigen::Matrix<double, 6, 6>::Identity()
  };
  Eigen::Matrix<double, 6, 6> cartesian_damping_target_{ Eigen::Matrix<double, 6, 6>::Identity() };
  Eigen::Matrix<double, 7, 1> damping_factors_{ Eigen::Matrix<double, 7, 1>::Ones() };

  Eigen::VectorXd q_;
  Eigen::VectorXd dq_;

  Eigen::MatrixXd jacobian_;

  Eigen::Matrix<double, 6, 1> error_;
  Eigen::Vector3d position_{ Eigen::Vector3d::Zero() };
  Eigen::Vector3d position_d_{ Eigen::Vector3d::Zero() };
  Eigen::Vector3d position_d_target_{ Eigen::Vector3d::Zero() };

  Eigen::Quaterniond orientation_{ Eigen::Quaterniond::Identity() };
  Eigen::Quaterniond orientation_d_{ Eigen::Quaterniond::Identity() };
  Eigen::Quaterniond orientation_d_target_{ Eigen::Quaterniond::Identity() };

  Eigen::Matrix<double, 6, 1> cartesian_wrench_{ Eigen::Matrix<double, 6, 1>::Zero() };
  Eigen::Matrix<double, 6, 1> cartesian_wrench_target_{ Eigen::Matrix<double, 6, 1>::Zero() };

  Eigen::VectorXd tau_c_;

  double update_frequency_{ 1000 };
  double filter_params_nullspace_config_{ 1.0 };
  double filter_params_stiffness_{ 1.0 };
  double filter_params_pose_{ 1.0 };
  double filter_params_wrench_{ 1.0 };

  double delta_tau_max_{ 1.0 };

private:
  // 根据刚度计算阻尼，规则为 2*sqrt(stiffness)
  double dampingRule(double stiffness) const;

  // 对所有刚度分量应用阻尼计算规则
  void applyDamping();

  // 设置控制器更新频率
  void setUpdateFrequency(double freq);

  // 设置滤波参数并校验取值范围
  void setFilterValue(double val, double* saved_val);

  // 对零空间关节配置进行比例滤波，逐步逼近目标
  void updateFilteredNullspaceConfig();

  // 对笛卡尔与零空间刚度进行比例滤波
  void updateFilteredStiffness();

  // 对末端参考位姿进行比例滤波
  void updateFilteredPose();

  // 对施加的笛卡尔力/力矩进行比例滤波
  void updateFilteredWrench();
};

}  // namespace cartesian_impedance_controller
