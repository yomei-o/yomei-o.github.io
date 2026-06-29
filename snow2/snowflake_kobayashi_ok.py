"""
雪結晶シミュレーション : 小林モデル (Kobayashi 1993) フェーズフィールド法
================================================================================

R. Kobayashi, "Modeling and numerical simulations of dendritic crystal growth",
Physica D 63 (1993) 410-423.

位相場 phi (0=液相/水蒸気, 1=固相/氷) と 無次元温度 T を連立で解く。

    tau * dphi/dt = eps^2 * lap(phi)
                    + d/dy( eps*eps' * dphi/dx )
                    - d/dx( eps*eps' * dphi/dy )
                    + phi(1-phi)(phi - 1/2 + m)

    dT/dt = lap(T) + K * dphi/dt          (潜熱の放出)

    m(T)  = (alpha/pi) * arctan( gamma * (T_eq - T) )   ← 駆動力(過冷却)

  6回対称(六方晶)の異方性:
    eps(theta)  = ebar * (1 + delta * cos( J*(theta - theta0) ))
    eps'(theta) = -ebar * J * delta * sin( J*(theta - theta0) )
    theta = atan2(dphi/dy, dphi/dx)


================================================================================
   ★ハマりどころ★  「六角形/四角の板になって枝が出ない」を解決した知見
================================================================================

このコードに行き着くまでに分かった、パラメータ以前の本質的な落とし穴:

(1) 離散化スキームが命。
    勾配項を「flux の発散 div(eps^2 grad phi)」としてまとめて差分し、さらに
    phi を [0,1] にクリップすると、数値拡散が強すぎて界面の不安定(分岐)が
    潰れ、ツルッとしたコンパクト円盤や六角板にしかならない。
    → 本コードのように項別(eps^2*lap + 交差項)で差分し、クリップしない方が
      Mullins-Sekerka 不安定がちゃんと立ち上がり、側枝つきの樹枝が出る。
    側枝は乱数ノイズが無くても自発的に発生する(これが小林モデルの肝)。

(2) theta0 を格子軸からずらす (例: 0.2 rad)。
    theta0=0 や pi/2 だと異方性の極大が正方格子の軸に揃い、格子の4回異方性と
    共鳴して結晶が四角くなりやすい。少し傾けると6回対称が素直に出る。

(3) 成長レジーム。
      過冷却 弱 (alpha 小)  → ファセットした六角「板」(枝なし)
      過冷却 中〜強(alpha 0.9)→ 6本腕の樹枝(側枝あり)  ← 雪結晶らしいのはココ
    枝を増やしたい/減らしたい時の効き方:
      delta↑  : 腕が細く鋭くなり側枝が増える (0.02〜0.06)
      K↑(潜熱): 腕の間の凍結が抑えられ枝が分離する。上げ過ぎると成長停止
      alpha↑  : 駆動力増。速いが上げ過ぎるとコンパクト塊化
      ノイズ  : 側枝の左右非対称・ランダムさを足したい時だけ少量(0〜0.02)

(4) 解像度。 dx を小さく、領域 N を大きく。粗いと格子異方性で六角→四角に化ける。

数値安定性: dt < dx^2/4 が目安(熱拡散のCFL)。本既定値 dt=1e-4 < 0.03^2/4=2.25e-4。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ----------------------------------------------------------------------------
# パラメータ (既定値で綺麗な6本腕の雪結晶が出る)
# ----------------------------------------------------------------------------
class Params:
    # 格子
    N      = 300          # 格子点数 (N x N)
    dx     = 0.03         # 格子間隔
    dt     = 1.0e-4       # 時間刻み (< dx^2/4 = 2.25e-4)

    # フェーズフィールド
    tau    = 3.0e-4       # 緩和時間 (小さいほど界面が速く動く)
    ebar   = 0.010        # 界面エネルギーの基準値 (界面厚みに対応)
    delta  = 0.04         # ★異方性強度。0.02=ふっくら 0.06=細く鋭い枝
    J      = 6            # ★対称性。6=雪(六方晶)。4 にすると正方デンドライト
    theta0 = 0.2          # ★異方性の基準角[rad]。格子軸から少しずらすのが重要

    # 駆動力 m(T)
    alpha  = 0.90         # ★駆動力(過冷却)の強さ。0.9で6本腕の樹枝
    gamma  = 10.0         # arctan の鋭さ
    T_eq   = 1.0          # 無次元平衡温度

    # 熱
    K      = 1.8          # ★潜熱係数。枝の分離度を決める。上げ過ぎると成長停止
    T_init = 0.0          # 初期過冷却温度 (低いほど過冷却が強い)

    # ノイズ (側枝は無くても出るが、ランダムさを足したいとき少量)
    noise_amp = 0.0       # 0〜0.02 程度

    # 実行
    seed_r = 5.0          # 初期核(中心の氷)の半径 [格子単位]
    nsteps = 4500         # 総ステップ数 (これ以上だと結晶が領域端に達する)
    plot_every = 50       # 何ステップごとに描画/保存するか
    rng_seed = 7


# ----------------------------------------------------------------------------
# 微分演算子 (周期境界 / 単純中心差分・5点ラプラシアン)
# ※ あえて単純な中心差分を使う。等方化した平滑な演算子だと分岐が潰れるため。
# ----------------------------------------------------------------------------
def grad_x(f, dx):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)

def grad_y(f, dx):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dx)

def laplacian(f, dx):
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0)
            + np.roll(f, 1, 1) + np.roll(f, -1, 1)
            - 4.0 * f) / (dx * dx)


# ----------------------------------------------------------------------------
# 初期化 : 中央に小さな氷の核
# ----------------------------------------------------------------------------
def initialize(p):
    phi = np.zeros((p.N, p.N))
    T   = np.full((p.N, p.N), p.T_init)
    c = p.N // 2
    yy, xx = np.mgrid[0:p.N, 0:p.N]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    phi[r < p.seed_r] = 1.0
    return phi, T


# ----------------------------------------------------------------------------
# 1ステップ更新 (Kobayashi/Biner スキーム)
# ----------------------------------------------------------------------------
def step(phi, T, p, rng):
    phi_x = grad_x(phi, p.dx)
    phi_y = grad_y(phi, p.dx)

    # 界面法線の角度に依存した異方性 eps(theta), eps'(theta)
    theta = np.arctan2(phi_y, phi_x)
    ang   = p.J * (theta - p.theta0)
    eps   = p.ebar * (1.0 + p.delta * np.cos(ang))
    eps_d = -p.ebar * p.J * p.delta * np.sin(ang)

    # 異方性の交差項   d/dy(eps eps' phi_x) - d/dx(eps eps' phi_y)
    a = eps * eps_d
    cross = grad_y(a * phi_x, p.dx) - grad_x(a * phi_y, p.dx)

    # 駆動力 + ダブルウェル (+ 任意のノイズ)
    m = (p.alpha / np.pi) * np.arctan(p.gamma * (p.T_eq - T))
    if p.noise_amp:
        m = m + p.noise_amp * (rng.random(phi.shape) - 0.5)
    react = phi * (1.0 - phi) * (phi - 0.5 + m)

    dphi = (cross + eps * eps * laplacian(phi, p.dx) + react) / p.tau

    phi_new = phi + p.dt * dphi          # ※クリップしない(分岐を残すため)
    T_new   = T + p.dt * (laplacian(T, p.dx) + p.K * dphi)
    return phi_new, T_new


# ----------------------------------------------------------------------------
# 実行 + 可視化
# ----------------------------------------------------------------------------
def run(p=Params(), save_gif=False, gif_path="snowflake.gif", show=True):
    rng = np.random.default_rng(p.rng_seed)
    phi, T = initialize(p)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()
    im = ax.imshow(phi, cmap="bone", vmin=0, vmax=1, interpolation="bilinear")
    title = ax.set_title("step 0")
    fig.tight_layout()

    state = {"phi": phi, "T": T}

    def update(frame):
        for _ in range(p.plot_every):
            state["phi"], state["T"] = step(state["phi"], state["T"], p, rng)
        im.set_data(state["phi"])
        title.set_text(f"step {frame * p.plot_every}  "
                       f"solid={np.mean(state['phi'] > 0.5):.2f}")
        return im, title

    nframes = p.nsteps // p.plot_every
    anim = FuncAnimation(fig, update, frames=nframes,
                         interval=30, blit=False, repeat=False)

    if save_gif:
        anim.save(gif_path, writer="pillow", fps=25)
        print(f"saved: {gif_path}")
    elif show:
        plt.show()
    return anim


def run_to_png(p=Params(), png_path="snowflake.png"):
    """アニメ無しで最後まで回し、結果を1枚のPNGに保存する(高速)。"""
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(p.rng_seed)
    phi, T = initialize(p)
    for i in range(p.nsteps):
        phi, T = step(phi, T, p, rng)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()
    ax.imshow(phi, cmap="bone", vmin=0, vmax=1, interpolation="bilinear")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    print(f"saved: {png_path}  (solid fraction = {np.mean(phi > 0.5):.3f})")
    return phi, T


if __name__ == "__main__":
    import sys
    if "--png" in sys.argv:
        run_to_png()
    elif "--gif" in sys.argv:
        run(save_gif=True)
    else:
        run()          # ライブアニメーション (matplotlib ウィンドウ)
