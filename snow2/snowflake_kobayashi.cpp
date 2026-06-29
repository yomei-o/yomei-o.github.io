// ============================================================================
//  雪結晶シミュレーション : 小林モデル (Kobayashi 1993) フェーズフィールド法
//  snowflake_kobayashi.py の C++ 移植版 (コンソールアプリ)
//
//  R. Kobayashi, "Modeling and numerical simulations of dendritic crystal
//  growth", Physica D 63 (1993) 410-423.
//
//  位相場 phi (0=液相/水蒸気, 1=固相/氷) と 無次元温度 T を連立で解く:
//
//      tau * dphi/dt = eps^2 * lap(phi)
//                      + d/dy( eps*eps' * dphi/dx )
//                      - d/dx( eps*eps' * dphi/dy )
//                      + phi(1-phi)(phi - 1/2 + m)
//      dT/dt = lap(T) + K * dphi/dt            (潜熱の放出)
//      m(T)  = (alpha/pi) * arctan( gamma * (T_eq - T) )
//
//  異方性 (6回対称):
//      eps(theta)  = ebar * (1 + delta * cos(J*(theta - theta0)))
//      eps'(theta) = -ebar * J * delta * sin(J*(theta - theta0))
//      theta = atan2(dphi/dy, dphi/dx)
//
//  境界条件は周期境界。結果は phi を bone カラーマップで PNG 保存する。
//
//  ビルド (Visual Studio 2022 の x64 Native Tools Command Prompt 等):
//      cl /O2 /EHsc /std:c++17 snowflake_kobayashi.cpp
//  実行:
//      snowflake_kobayashi.exe            -> snowflake.png を出力
//      snowflake_kobayashi.exe out.png    -> 出力先を指定
// ============================================================================

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ----------------------------------------------------------------------------
// パラメータ (既定値で綺麗な6本腕の雪結晶が出る)
// ----------------------------------------------------------------------------
struct Params {
    // 格子 (細かい枝を解像するため細グリッド・広領域)
    int    N      = 520;        // 格子点数 (N x N)
    double dx     = 0.024;      // 格子間隔 (小さいほど細い構造を解像)
    double dt     = 1.1e-4;     // 時間刻み (< dx^2/4 = 1.44e-4)

    // フェーズフィールド
    double tau    = 3.0e-4;     // 緩和時間 (小さいほど界面が速く動く)
    double ebar   = 0.008;      // 界面エネルギー(≒毛細管長)。小さいほど幹が細い
    double delta  = 0.03;       // 異方性強度。小さめだと針状で細い
    int    J      = 6;          // 対称性。6=雪(六方晶)。4 にすると正方デンドライト
    double theta0 = 0.2;        // 異方性の基準角[rad]。格子軸から少しずらす

    // 駆動力 m(T)
    double alpha  = 0.95;       // 駆動力(過冷却)。高いほど側枝が多く細かい
    double gamma  = 10.0;       // arctan の鋭さ
    double T_eq   = 1.0;        // 無次元平衡温度

    // 熱
    double K      = 2.2;        // 潜熱係数。高いほど枝が細く分離する
    double T_init = 0.0;        // 初期過冷却温度

    // 実行
    double seed_r = 4.0;        // 初期核(中心の氷)の半径 [格子単位]
    int    nsteps = 8000;       // 総ステップ数 (長いほど腕が伸びる)
    int    report_every = 500;  // 進捗表示間隔
};

// ----------------------------------------------------------------------------
// 2次元場 (周期境界アクセス付き)
// ----------------------------------------------------------------------------
struct Field {
    int N = 0;
    std::vector<double> a;
    Field() {}
    Field(int n, double v = 0.0) : N(n), a((size_t)n * n, v) {}
    inline double&       operator()(int i, int j)       { return a[(size_t)i * N + j]; }
    inline const double& operator()(int i, int j) const { return a[(size_t)i * N + j]; }
};

// 周期境界用のインデックス折り返し
static inline int wrap(int k, int N) {
    if (k < 0)  return k + N;
    if (k >= N) return k - N;
    return k;
}

// ----------------------------------------------------------------------------
// bone カラーマップ (matplotlib 互換)
//   区分線形:  0.0 -> (0,0,0)
//             3/8 -> 反映済み (下の制御点参照)
//   bone(x) = mix of grayscale with a slight blue tint, per matplotlib 定義。
//     r = x                               (x<3/4)  / (11x-7)/4  (x>=3/4)
//     g = 7x/8                            (x<3/8)  / (7x-3)/8 ... 実定義に従う
//   ここでは matplotlib の _bone_data 区分線形をそのまま実装する。
// ----------------------------------------------------------------------------
static void bone_colormap(double x, uint8_t& R, uint8_t& G, uint8_t& B) {
    if (x < 0.0) x = 0.0;
    if (x > 1.0) x = 1.0;
    double r, g, b;
    // --- matplotlib _bone_data の正確な区分線形 (x in [0,1]) ---
    // red:   (0,0,0) (0.746032, 0.652778) (1,1)
    if (x <= 0.746032) r = 0.652778 / 0.746032 * x;
    else               r = 0.652778 + (1.0 - 0.652778) / (1.0 - 0.746032) * (x - 0.746032);
    // green: (0,0,0) (0.365079,0.319444) (0.746032,0.777778) (1,1)
    if (x <= 0.365079)
        g = 0.319444 / 0.365079 * x;
    else if (x <= 0.746032)
        g = 0.319444 + (0.777778 - 0.319444) / (0.746032 - 0.365079) * (x - 0.365079);
    else
        g = 0.777778 + (1.0 - 0.777778) / (1.0 - 0.746032) * (x - 0.746032);
    // blue:  (0,0,0) (0.365079,0.444444) (1,1)
    if (x <= 0.365079)
        b = 0.444444 / 0.365079 * x;
    else
        b = 0.444444 + (1.0 - 0.444444) / (1.0 - 0.365079) * (x - 0.365079);

    auto to8 = [](double v) -> uint8_t {
        v = std::clamp(v, 0.0, 1.0);
        int iv = (int)std::lround(v * 255.0);
        return (uint8_t)std::clamp(iv, 0, 255);
    };
    R = to8(r); G = to8(g); B = to8(b);
}

// ----------------------------------------------------------------------------
// 1ステップ更新 (Kobayashi/Biner スキーム, 周期境界・中心差分)
//   ※ あえて単純な中心差分を使う。等方化した平滑な演算子だと分岐が潰れるため。
//   ※ phi はクリップしない(分岐を残すため)。
// ----------------------------------------------------------------------------
static void step(const Field& phi, const Field& T,
                 Field& phi_new, Field& T_new,
                 std::vector<double>& a,   // 作業配列: a = eps*eps' (各点)
                 std::vector<double>& ax,  // 作業配列: a * phi_x
                 std::vector<double>& ay,  // 作業配列: a * phi_y
                 const Params& p) {
    const int N = p.N;
    const double dx = p.dx;
    const double inv2dx = 1.0 / (2.0 * dx);
    const double invdx2 = 1.0 / (dx * dx);

    // --- 1パス目: 各点で勾配・異方性を計算し a, a*phi_x, a*phi_y を保存 ---
    for (int i = 0; i < N; ++i) {
        int im = wrap(i - 1, N), ip = wrap(i + 1, N);
        for (int j = 0; j < N; ++j) {
            int jm = wrap(j - 1, N), jp = wrap(j + 1, N);
            double phi_x = (phi(i, jp) - phi(i, jm)) * inv2dx;   // d/dx (axis=1)
            double phi_y = (phi(ip, j) - phi(im, j)) * inv2dx;   // d/dy (axis=0)

            double theta = std::atan2(phi_y, phi_x);
            double ang   = p.J * (theta - p.theta0);
            double eps   = p.ebar * (1.0 + p.delta * std::cos(ang));
            double eps_d = -p.ebar * p.J * p.delta * std::sin(ang);

            size_t idx = (size_t)i * N + j;
            a[idx]  = eps * eps_d;
            ax[idx] = a[idx] * phi_x;
            ay[idx] = a[idx] * phi_y;
        }
    }

    // --- 2パス目: 交差項 (a*phi_x, a*phi_y の発散) + eps^2*lap + 反応項 ---
    for (int i = 0; i < N; ++i) {
        int im = wrap(i - 1, N), ip = wrap(i + 1, N);
        for (int j = 0; j < N; ++j) {
            int jm = wrap(j - 1, N), jp = wrap(j + 1, N);
            size_t idx = (size_t)i * N + j;

            // cross = d/dy(a*phi_x) - d/dx(a*phi_y)
            double cross =
                (ax[(size_t)ip * N + j] - ax[(size_t)im * N + j]) * inv2dx
              - (ay[(size_t)i * N + jp] - ay[(size_t)i * N + jm]) * inv2dx;

            // eps(theta) をこの点で再計算 (eps^2*lap(phi) のため)
            double phi_x = (phi(i, jp) - phi(i, jm)) * inv2dx;
            double phi_y = (phi(ip, j) - phi(im, j)) * inv2dx;
            double theta = std::atan2(phi_y, phi_x);
            double ang   = p.J * (theta - p.theta0);
            double eps   = p.ebar * (1.0 + p.delta * std::cos(ang));

            double lap_phi = (phi(ip, j) + phi(im, j)
                            + phi(i, jp) + phi(i, jm)
                            - 4.0 * phi(i, j)) * invdx2;

            double Tij = T(i, j);
            double m = (p.alpha / M_PI) * std::atan(p.gamma * (p.T_eq - Tij));
            double pij = phi(i, j);
            double react = pij * (1.0 - pij) * (pij - 0.5 + m);

            double dphi = (cross + eps * eps * lap_phi + react) / p.tau;

            double lap_T = (T(ip, j) + T(im, j)
                          + T(i, jp) + T(i, jm)
                          - 4.0 * Tij) * invdx2;

            phi_new.a[idx] = pij + p.dt * dphi;            // クリップしない
            T_new.a[idx]   = Tij + p.dt * (lap_T + p.K * dphi);
        }
    }
}

int main(int argc, char** argv) {
    Params p;
    std::string png_path = (argc > 1) ? argv[1] : "snowflake.png";

    const int N = p.N;
    Field phi(N, 0.0), T(N, p.T_init);
    Field phi_new(N, 0.0), T_new(N, 0.0);
    std::vector<double> a((size_t)N * N), ax((size_t)N * N), ay((size_t)N * N);

    // 初期化: 中央に小さな氷の核
    int c = N / 2;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double r = std::sqrt((double)(j - c) * (j - c) + (double)(i - c) * (i - c));
            if (r < p.seed_r) phi(i, j) = 1.0;
        }

    std::printf("Kobayashi snowflake (phase-field)  N=%d  nsteps=%d  dt=%.2e\n",
                N, p.nsteps, p.dt);
    std::printf("dx^2/4 = %.3e  (dt must be smaller)\n", p.dx * p.dx / 4.0);

    for (int s = 0; s < p.nsteps; ++s) {
        step(phi, T, phi_new, T_new, a, ax, ay, p);
        std::swap(phi.a, phi_new.a);
        std::swap(T.a,   T_new.a);

        if (p.report_every > 0 && (s + 1) % p.report_every == 0) {
            // 固相率
            size_t solid = 0;
            for (double v : phi.a) if (v > 0.5) ++solid;
            double frac = (double)solid / phi.a.size();
            std::printf("  step %5d / %d   solid fraction = %.3f\n",
                        s + 1, p.nsteps, frac);
            std::fflush(stdout);
        }
    }

    // PNG 出力 (bone カラーマップ, vmin=0 vmax=1)
    std::vector<uint8_t> img((size_t)N * N * 3);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double v = phi(i, j);
            uint8_t R, G, B;
            bone_colormap(v, R, G, B);
            size_t k = ((size_t)i * N + j) * 3;
            img[k + 0] = R; img[k + 1] = G; img[k + 2] = B;
        }

    if (stbi_write_png(png_path.c_str(), N, N, 3, img.data(), N * 3)) {
        std::printf("saved: %s  (%d x %d)\n", png_path.c_str(), N, N);
        return 0;
    } else {
        std::fprintf(stderr, "ERROR: failed to write %s\n", png_path.c_str());
        return 1;
    }
}
