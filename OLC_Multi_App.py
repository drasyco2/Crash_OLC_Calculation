import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

# 设置网页基础配置（宽屏布局）
st.set_page_config(page_title="OLC 多曲线独立初速度分析工具", layout="wide")

# 支持中文显示的安全字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 页面头部设计 ====================
st.title("🚗 OLC 多曲线独立初速度分析工具")
st.markdown("""
本工具支持为**每条上传的曲线设置不同的初始速度**。上传文件后，您可以在“初始速度配置表”中直接修改对应文件的车速，系统将自动实时重算并更新对比图表。
***
""")

# ==================== 左侧侧边栏：其他通用参数 ====================
st.sidebar.header("🛠️ 约束与位移阈值设置")

free_flight_displacement = st.sidebar.number_input(
    "自由飞行位移 t1 阈值 (m):", 
    value=0.065, 
    format="%.3f",
    help="用于计算 t1 边界的自由飞行位移，默认 0.065m"
)
ideal_restraint_displacement = st.sidebar.number_input(
    "理想约束位移 delta_s 阈值 (m):", 
    value=0.235, 
    format="%.3f",
    help="用于计算 t2 边界的理想约束位移，默认 0.235m"
)

g = 9.81

# ==================== 主界面：多文件上传 ====================
st.subheader("📂 第一步：上传数据文件（支持多选）")
uploaded_files = st.file_uploader(
    "请选择一个或多个 CSV 文件（第一列为时间/s，第二列为加速度）", 
    type=["csv"],
    accept_multiple_files=True,
    help="您可以按住 Ctrl 或 Command 键批量选择多个 CSV 文件上传"
)

# ==================== 核心逻辑处理 ====================
if uploaded_files:
    st.markdown("---")
    st.subheader("⚙️ 第二步：个性化初始速度设置")
    st.info("💡 提示：您可以在下方的表格中，直接双击 **“初始速度(km/h)”** 列来修改对应文件的速度，修改后下方结果会自动刷新！")

    # 1. 初始化或获取会话状态中的速度配置表
    # 这样能保证用户修改数字时，表格状态不会被轻易重置
    file_names = [f.name for f in uploaded_files]
    
    if "velocity_df" not in st.session_state or set(st.session_state.velocity_df["文件名称"]) != set(file_names):
        # 如果是新上传的文件，默认初速度给 56.0
        default_data = {"文件名称": file_names, "初始速度(km/h)": [56.0] * len(file_names)}
        st.session_state.velocity_df = pd.DataFrame(default_data)

    # 2. 渲染可直接在网页上编辑的 Excel 样式表格
    edited_df = st.data_editor(
        st.session_state.velocity_df,
        use_container_width=True,
        hide_index=True,
        disabled=["文件名称"], # 文件名锁死，不允许修改
        column_config={
            "初始速度(km/h)": st.column_config.NumberColumn(
                "初始速度(km/h)",
                help="双击修改此处的速度值",
                min_value=0.1,
                max_value=200.0,
                step=1.0,
                format="%.1f"
            )
        }
    )
    # 将用户修改后的最新数据存回 session_state
    st.session_state.velocity_df = edited_df

    # 将编辑后的速度构建成字典，方便后面查询，例如：{"test1.csv": 56.0, "test2.csv": 48.0}
    velocity_dict = dict(zip(edited_df["文件名称"], edited_df["初始速度(km/h)"]))

    # 3. 开始执行批量 OLC 计算
    results_list = []
    plot_data = {}

    for uploaded_file in uploaded_files:
        try:
            # 读取当前文件的速度配置
            v_kmh = velocity_dict.get(uploaded_file.name, 56.0)
            v_ms = v_kmh * 1000 / 3600 # 转换为 m/s

            # 读取CSV
            data = pd.read_csv(uploaded_file, header=None)
            if data.shape[1] < 2:
                st.error(f"❌ 错误：文件 `{uploaded_file.name}` 列数不足，已自动跳过。")
                continue
                
            time_raw = data.iloc[:, 0].values
            acc_raw = data.iloc[:, 1].values
            
            # --- OLC 核心计算部分 ---
            velocity_t = cumulative_trapezoid(acc_raw, time_raw, initial=0)
            multi_g = abs(velocity_t[-1] / v_ms)

            if multi_g < 0.125:  # 1/8 逻辑
                velocity = v_ms - g * velocity_t
                if velocity[-1] > v_ms:
                    velocity = v_ms + g * velocity_t
            else:
                velocity = v_ms - velocity_t
                if velocity[-1] > v_ms:
                    velocity = v_ms + velocity_t

            displacement = cumulative_trapezoid(velocity, time_raw, initial=0)

            # 计算 t1
            integral_V0 = v_ms * time_raw
            integral_V = cumulative_trapezoid(velocity, time_raw, initial=0)
            relative_displacement = integral_V0 - integral_V

            t1_indices = np.where(relative_displacement >= free_flight_displacement)[0]
            t1_index = t1_indices[0] if len(t1_indices) > 0 else len(time_raw) - 1
            t1 = time_raw[t1_index]

            # 迭代寻找 t2
            tolerances = [1e-4, 1e-3, 1e-2]
            found_valid = False
            OLC, t2 = 0, 0
            time_calc, vel_calc, disp_calc = time_raw.copy(), velocity.copy(), displacement.copy()

            for tolerance in tolerances:
                t2_index = t1_index
                delta_s_last = None

                while t2_index < len(time_calc):
                    t2_curr = time_calc[t2_index]
                    V2_curr = vel_calc[t2_index]
                    s_MPDB_t1 = disp_calc[t1_index]
                    s_MPDB_t2 = disp_calc[t2_index]
                    s_MPDB = s_MPDB_t2 - s_MPDB_t1
                    s_IR = (v_ms + V2_curr) * (t2_curr - t1) / 2
                    delta_s = s_IR - s_MPDB

                    if abs(ideal_restraint_displacement - delta_s) <= tolerance:
                        OLC = (1 / g) * (v_ms - V2_curr) / (t2_curr - t1)
                        t2 = t2_curr
                        found_valid = True
                        break
                    delta_s_last = delta_s
                    t2_index += 1
                if found_valid:
                    break

            # 数据外扩逻辑
            if not found_valid and delta_s_last is not None and delta_s_last < ideal_restraint_displacement:
                time_step = time_calc[1] - time_calc[0]
                velocity_last = vel_calc[-1]

                while delta_s_last < ideal_restraint_displacement:
                    time_calc = np.append(time_calc, time_calc[-1] + time_step)
                    vel_calc = np.append(vel_calc, velocity_last)
                    disp_calc = cumulative_trapezoid(vel_calc, time_calc, initial=0)

                    new_t2_index = t1_index
                    while new_t2_index < len(time_calc):
                        t2_curr = time_calc[new_t2_index]
                        V2_curr = vel_calc[new_t2_index]
                        s_MPDB_t1 = disp_calc[t1_index]
                        s_MPDB_t2 = disp_calc[new_t2_index]
                        s_MPDB = s_MPDB_t2 - s_MPDB_t1
                        s_IR = (v_ms + V2_curr) * (t2_curr - t1) / 2
                        delta_s = s_IR - s_MPDB

                        if delta_s > ideal_restraint_displacement:
                            OLC = (1 / g) * (v_ms - V2_curr) / (t2_curr - t1)
                            t2 = t2_curr
                            found_valid = True
                            break
                        new_t2_index += 1
                    if found_valid:
                        break

            # 保存当前文件的数据以便后续绘图
            plot_data[uploaded_file.name] = {
                'time': time_calc,
                'velocity': vel_calc,
                'acc': acc_raw,
                'time_raw': time_raw,
                't1': t1,
                't2': t2 if found_valid else None,
                'v_init_ms': v_ms,
                'v_init_kmh': v_kmh
            }

            # 汇总结果
            results_list.append({
                "📄 文件名称": uploaded_file.name,
                "🚗 输入初速度(km/h)": v_kmh,
                "🎯 OLC 值 (g)": round(OLC, 4) if OLC != 0 else "计算失败",
                "⏱️ t1 时刻 (s)": round(t1, 4),
                "⏱️ t2 时刻 (s)": round(t2, 4) if (found_valid and t2 != 0) else "未找到"
            })

        except Exception as e:
            st.error(f"❌ 解析文件 `{uploaded_file.name}` 时出错: {str(e)}")

    # ==================== 展示汇总结果表格 ====================
    st.markdown("---")
    st.subheader("📊 第三步：OLC 计算结果汇总对比")
    
    if results_list:
        summary_df = pd.DataFrame(results_list)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # ==================== 绘制叠加对比图表 ====================
        st.markdown("---")
        st.subheader("📈 第四步：曲线叠加对比图")
        
        tab1, tab2 = st.tabs(["💧 速度曲线叠加对比 (Velocity)", "⚡ 加速度曲线叠加对比 (Acceleration)"])
        
        # 1. 速度曲线叠加（初速度不同时，各曲线起始高度会错开，非常直观）
        with tab1:
            fig_v, ax_v = plt.subplots(figsize=(11, 5.5))
            
            for file_name, d in plot_data.items():
                # 绘制各个文件的速度曲线
                line, = ax_v.plot(d['time'], d['velocity'], linewidth=2, 
                                  label=f"{file_name} (V0={d['v_init_kmh']}km/h)")
                # 绘制各自的初始速度基准线（虚线）
                ax_v.axhline(y=d['v_init_ms'], color=line.get_color(), linestyle='--', alpha=0.4)
                
                # 标记各自的边界时刻
                if d['t1']:
                    ax_v.axvline(x=d['t1'], color=line.get_color(), linestyle=':', alpha=0.5)
                if d['t2']:
                    ax_v.axvline(x=d['t2'], color=line.get_color(), linestyle='-.', alpha=0.5)
                    
            ax_v.set_xlabel('Time(s)', fontsize=11)
            ax_v.set_ylabel('Velocity (m/s)', fontsize=11)
            ax_v.set_title('Multi-velocity Curves Comparison', fontsize=13, pad=12)
            ax_v.grid(True, alpha=0.3)
            ax_v.legend(loc='upper right', bbox_to_anchor=(1.25, 1)) 
            st.pyplot(fig_v)
            
        # 2. 加速度曲线叠加
        with tab2:
            fig_a, ax_a = plt.subplots(figsize=(11, 5.5))
            
            for file_name, d in plot_data.items():
                ax_a.plot(d['time_raw'], d['acc'], linewidth=1.5, alpha=0.8, label=f"{file_name}")
                
            ax_a.set_xlabel('Time(s)', fontsize=11)
            ax_a.set_ylabel('Acceleration(m/s²)', fontsize=11)
            ax_a.set_title('Multi-acceleration Curves Comparison', fontsize=13, pad=12)
            ax_a.grid(True, alpha=0.3)
            ax_a.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
            st.pyplot(fig_a)

        # ==================== 批量结果导出 ====================
        st.markdown("---")
        st.subheader("📥 第五步：导出汇总报告")
        csv_buffer = summary_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="点击下载带有OLC的结果汇总表 (CSV格式)",
            data=csv_buffer,
            file_name="OLC_Multi_Velocity_Results.csv",
            mime="text/csv"
        )
