import streamlit as st
import pandas as pd
import numpy as np
import pulp

st.set_page_config(page_title="集合駅 最適化デモ（PuLP × Streamlit）", layout="wide")

st.title("集合駅 最適化デモ")
st.caption("PuLPの0-1最適化をStreamlitでWeb化（単一スライダー w：時間 w / 運賃 1-w）")

# --- 既定データ ---
default_I = ['錦糸町', '大崎', '池袋', '府中本町', '神田', '千駄ヶ谷' ,'渋谷', '登戸']
default_J = ['新宿', '渋谷', '池袋', '北千住', '東京', '高田馬場', '新橋', '品川']

default_Time = [
    [25, 29, 32, 13,  8, 31, 13, 17],
    [11,  5, 19, 42, 17, 20, 12,  3],
    [ 8, 10,  0, 20, 16,  4, 28, 28],
    [26, 50, 55, 80, 58, 51, 66, 63],
    [18, 28, 25, 28,  5, 25, 10, 18],
    [ 9, 13, 20, 42, 23, 16, 31, 26],
    [10,  0, 18, 44, 27, 14, 20, 16],
    [22, 25, 33, 60, 50, 29, 48, 48],
]

# ※ Cost 6行目は8列に調整済み
default_Cost = [
    [230, 260, 230, 360, 170, 230, 180, 230],
    [180, 170, 210, 320, 180, 210, 180, 150],
    [170, 180,   0, 230, 210, 150, 210, 280],
    [460, 460, 580, 860, 660, 490, 660, 580],
    [180, 210, 210, 180, 150, 210, 170, 180],
    [150, 170, 180, 390, 180, 180, 170, 210],
    [170,   0, 180, 260, 210, 180, 180, 180],
    [270, 370, 440, 490, 440, 420, 440, 320],
]

# --- サイドバー：設定 ---
with st.sidebar:
    st.header("設定")
    time_limit = st.number_input("各出発駅の時間上限（分）", min_value=1, value=30, step=1)
    cost_limit = st.number_input("各出発駅の運賃上限（円）", min_value=1, value=999, step=1)
    st.markdown("---")
    w = st.slider("重み w（時間 = w, 運賃 = 1 - w）", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    st.caption(f"現在の重み：時間 = {w:.2f}, 運賃 = {1-w:.2f}（合計=1.00）")
    st.markdown("---")
    solve_btn = st.button("最適化を実行", type="primary")

# --- メイン：データ編集UI ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("出発駅（I）")
    I_df = st.data_editor(
        pd.DataFrame({"出発駅": default_I}),
        num_rows="dynamic",
        use_container_width=True,
        key="I_editor",
    )
with col2:
    st.subheader("集合候補（J）")
    J_df = st.data_editor(
        pd.DataFrame({"集合駅候補": default_J}),
        num_rows="dynamic",
        use_container_width=True,
        key="J_editor",
    )

I = I_df["出発駅"].astype(str).tolist()
J = J_df["集合駅候補"].astype(str).tolist()
Dep, Gat = len(I), len(J)

st.markdown("---")
st.subheader("所要時間行列 Time（分）")
Time_df = st.data_editor(
    pd.DataFrame(default_Time, index=I, columns=J),
    num_rows=Dep, use_container_width=True, key="Time_editor",
)

st.subheader("運賃行列 Cost（円）")
Cost_df = st.data_editor(
    pd.DataFrame(default_Cost, index=I, columns=J),
    num_rows=Dep, use_container_width=True, key="Cost_editor",
)

# --- バリデーション ---
def validate_matrix(df: pd.DataFrame, name: str) -> tuple[bool, str]:
    ok = True
    msg = ""
    if df.shape != (Dep, Gat):
        ok = False
        msg = f"{name} の形が不正です。期待: ({Dep}, {Gat}) / 実際: {df.shape}"
    elif not np.isfinite(df.values).all():
        ok = False
        msg = f"{name} に数値以外/NaN/infが含まれています。"
    return ok, msg

valid_time, time_msg = validate_matrix(Time_df, "Time")
valid_cost, cost_msg = validate_matrix(Cost_df, "Cost")

if not valid_time:
    st.error(time_msg)
if not valid_cost:
    st.error(cost_msg)

# --- 最適化 ---
def solve(Time_mat, Cost_mat, I, J, time_limit, cost_limit, w):
    problem = pulp.LpProblem("example", pulp.LpMinimize)

    # 変数 x_j ∈ {0,1}
    x = {j_idx: pulp.LpVariable(f"x{j_idx}", cat="Binary") for j_idx in range(len(J))}

    # 制約：候補はちょうど一つ
    problem += pulp.lpSum(x[j] for j in range(len(J))) == 1

    # 制約：各出発駅 i に対し Time_i, Cost_i の上限
    for i in range(len(I)):
        problem += pulp.lpSum(x[j] * Time_mat[i][j] for j in range(len(J))) <= time_limit
        problem += pulp.lpSum(x[j] * Cost_mat[i][j] for j in range(len(J))) <= cost_limit

    # 列ごとの範囲（max-min）を事前計算
    time_ranges = []
    cost_ranges = []
    for j in range(len(J)):
        col_time = [Time_mat[i][j] for i in range(len(I))]
        col_cost = [Cost_mat[i][j] for i in range(len(I))]
        time_ranges.append(max(col_time) - min(col_time))
        cost_ranges.append(max(col_cost) - min(col_cost))

    # 重み：時間 = w, 運賃 = 1 - w（合計1）
    w_time = w
    w_cost = 1.0 - w

    # 目的関数：w_time * range(Time[:,j]) + w_cost * range(Cost[:,j])
    problem += pulp.lpSum((w_time * time_ranges[j] + w_cost * cost_ranges[j]) * x[j]
                          for j in range(len(J)))

    # ソルバー（CBC）で解く
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = problem.solve(solver)

    return problem, x, status, time_ranges, cost_ranges, w_time, w_cost

if solve_btn:
    if valid_time and valid_cost and Dep > 0 and Gat > 0:
        Time_mat = Time_df.values.tolist()
        Cost_mat = Cost_df.values.tolist()

        try:
            problem, x, status, time_ranges, cost_ranges, w_time, w_cost = solve(
                Time_mat, Cost_mat, I, J, time_limit, cost_limit, w
            )
            st.markdown("---")
            st.subheader("結果")

            st.write("**解状態**：", pulp.LpStatus[status])
            st.write(f"**重み**： 時間 = **{w_time:.2f}**, 運賃 = **{w_cost:.2f}**（合計=1.00）")
            if pulp.LpStatus[status] == "Optimal":
                chosen_j = None
                for j in range(Gat):
                    if x[j].value() == 1:
                        chosen_j = j
                        break
                if chosen_j is not None:
                    st.success(f"最適な集合駅は **{J[chosen_j]}** です。")
                    st.write(f"目的関数値： **{problem.objective.value():,.3f}**")

                    diag = pd.DataFrame({
                        "集合駅候補": J,
                        "Time範囲": time_ranges,
                        "Cost範囲": cost_ranges,
                        "加重合計": [w_time * time_ranges[j] + w_cost * cost_ranges[j] for j in range(Gat)],
                        "選択": ["◯" if j == chosen_j else "" for j in range(Gat)],
                    })
                    st.dataframe(diag, use_container_width=True)
                else:
                    st.warning("解は最適ですが、選択列が取得できませんでした。")
            elif pulp.LpStatus[status] in {"Infeasible", "Unbounded"}:
                st.error("実行不能または非有界です。制約（時間上限/運賃上限）を緩める/データを見直してください。")
            else:
                st.warning("最適解が得られませんでした。ソルバーのログやデータを確認してください。")

        except Exception as e:
            st.exception(e)
    else:
        st.warning("データの形や値を修正してから再実行してください。")
