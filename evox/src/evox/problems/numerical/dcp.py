import torch

from evox.core import Problem
from evox.operators.sampling import grid_sampling, uniform_sampling
from evox.operators.selection import non_dominate_rank
import numpy as np

class DCP(Problem):
    """
    Base class for DTLZ test suite problems in multi-objective optimization.

    Inherit this class to implement specific DTLZ problem variants.

    :param d: Number of decision variables.
    :param m: Number of objectives.
    :param ref_num: Number of reference points used in the problem.
    """

    def __init__(self, d: int = 10, m: int = 2, ref_num: int = 1000, taut: int = 20, nt: int=10, maxG: int = 100):
        """Override the setup method to initialize the parameters"""
        super().__init__()
        self.d = d
        self.m = m
        self.ref_num = ref_num
        self.sample, _ = uniform_sampling(self.ref_num * self.m, self.m)  # Assuming UniformSampling is defined
        self.device = self.sample.device
        self.taut = taut
        self.nt = nt
        self.lb = torch.zeros(self.d)
        self.ub = torch.ones(self.d)
        self.maxG = maxG



    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to evaluate the objective values for given decision variables.

        :param X: A tensor of shape (n, d), where n is the number of solutions and d is the number of decision variables.
        :return: A tensor of shape (n, m) representing the objective values for each solution.
        """
        raise NotImplementedError()

    def pf(self):
        """
        Return the Pareto front for the problem.

        :return: A tensor representing the Pareto front.
        """
        f = self.sample / 2
        return f


class DCP1(DCP):
    def __init__(self, d: int = 10, m: int = 2, ref_num: int = 1000, taut: int = 20, nt: int=10, maxG: int = 100):
        super().__init__(d, m, ref_num, taut, nt, maxG)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        t = torch.floor(torch.tensor(self.Dtime / self.taut)) / self.nt
        G = torch.abs(torch.sin(0.5 * torch.pi * t))
        g = 1 + torch.sum((X[:, 1:] - G) ** 2, dim=1, keepdim=True)
        obj1 = g * X[:, 0:1] + G
        obj2 = g * (1 - X[:, 0:1]) + G
        PopObj = torch.cat([obj1, obj2], dim=1)
        PopObj[PopObj < 1e-18] = 0.0
        angle_rad = torch.tensor(-0.15 * torch.pi)
        cos_val = torch.cos(angle_rad)
        sin_val = torch.sin(angle_rad)
        angle_comp_1 = cos_val * PopObj[:, 0:1]
        angle_comp_2 = sin_val * PopObj[:, 1:2]
        inner_sin_arg = 5 * torch.pi * (angle_comp_2 + angle_comp_1)
        PopCon = (2 * torch.sin(inner_sin_arg)) ** 6 - PopObj[:, 1:2] * cos_val + PopObj[:, 0:1] * sin_val
        return PopObj, PopCon

    def pf(self) -> torch.Tensor:
        device = self.device
        dtype = torch.float32

        # 计算时间序列tt和H
        fe_values = torch.arange(0, self.maxG + 1, device=device, dtype=dtype)
        tt = torch.floor(fe_values  / self.taut) / self.nt
        H = torch.sin(0.01 * torch.pi * tt)
        H = torch.unique(torch.round(H * 1e6) / 1e6)

        x = torch.linspace(0, 1, 500, device=device, dtype=dtype).reshape(-1, 1)
        addtion1 = torch.arange(0.001, 0.401, 0.001, device=device, dtype=dtype).reshape(-1, 1)

        all_R = []

        for i in range(len(H)):
            t = i / self.nt
            G = torch.abs(torch.sin(0.5 * torch.pi * torch.tensor(t, device=device, dtype=dtype)))

            P1 = x + G
            P2 = 1 - x + G

            addtion2 = torch.full_like(addtion1, torch.min(P1))
            addtion3 = addtion1 + torch.max(P1)

            P1 = torch.cat([P1, addtion2, addtion3], dim=0)
            P2 = torch.cat([P2, addtion3, addtion2], dim=0)

            sin_term = torch.sin(torch.tensor(-0.15 * torch.pi, device=device, dtype=dtype))
            cos_term = torch.cos(torch.tensor(-0.15 * torch.pi, device=device, dtype=dtype))

            Con = (2 * torch.sin(5 * torch.pi * (sin_term * P2 + cos_term * P1))) ** 6 - \
                  cos_term * P2 + sin_term * P1

            feasible_mask = Con.squeeze() <= 0
            P1_feasible = P1[feasible_mask]
            P2_feasible = P2[feasible_mask]

            if P1_feasible.numel() > 0:
                R = torch.cat([P1_feasible, P2_feasible], dim=1)
                rank = non_dominate_rank(R)
                R = R[rank == 0]
                all_R.append(R)

        return all_R

class DCP2(DCP):
    def __init__(self, d: int = 10, m: int = 2, ref_num: int = 1000, taut: int = 20, nt: int=10, maxG: int = 100):
        super().__init__(d, m, ref_num, taut, nt, maxG)
        self.lb = torch.tensor([0.0] + [-1.0] * (self.d - 1))
        self.ub = torch.tensor([1.0] * self.d)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        t = torch.floor(torch.tensor(self.Dtime / self.taut)) / self.nt
        G_obj = torch.sin(0.5 * torch.pi * t)
        x_minus_G = X[:, 1:] - G_obj
        g = 1 + torch.sum(x_minus_G**2 + torch.sin(0.5 * torch.pi * x_minus_G)**2, dim=1, keepdim=True)

        common_term = 0.25 * G_obj * torch.sin(torch.pi * X[:, 0:1])
        obj1 = g * (X[:, 0:1] + common_term)
        obj2 = g * (1 - X[:, 0:1] + common_term)

        PopObj = torch.cat([obj1, obj2], dim=1)
        PopObj[PopObj < 1e-18] = 0.0

        con1 = -(4. * PopObj[:, 0:1] + PopObj[:, 1:2] - 1) * \
               (0.3 * PopObj[:, 0:1] + PopObj[:, 1:2] - 0.3)

        # PopCon(:,2)=(1.85-PopObj(:,1)-PopObj(:,2) - (0.3.*sin(3*pi.*(PopObj(:,2)-PopObj(:,1)))).^2).*(PopObj(:,1)+PopObj(:,2)-1.3);
        obj_sum = PopObj[:, 0:1] + PopObj[:, 1:2]
        obj_diff = PopObj[:, 1:2] - PopObj[:, 0:1]
        con2 = (1.85 - obj_sum - (0.3 * torch.sin(3 * torch.pi * obj_diff)) ** 2) * \
               (obj_sum - 1.3)

        PopCon = torch.cat([con1, con2], dim=1)

        return PopObj, PopCon

    def pf(self) -> list:
        device = self.device
        dtype = torch.float32

        # 计算时间序列tt和H
        fe_values = torch.arange(0, self.maxG + 1, device=device, dtype=dtype)
        tt = torch.floor(fe_values  / self.taut) / self.nt
        H = torch.sin(0.01 * torch.pi * tt)
        H = torch.unique(torch.round(H * 1e6) / 1e6)

        # 创建pf2网格
        x2 = torch.arange(0, 2.001, 0.001, device=device, dtype=dtype)
        n_x2 = len(x2)
        pf2_col1 = x2.repeat(n_x2)
        pf2_col2 = x2.repeat_interleave(n_x2)
        pf2 = torch.stack([pf2_col1, pf2_col2], dim=1)

        # 筛选pf2
        cc = 1.85 - pf2[:, 1] - pf2[:, 0] - \
             (0.3 * torch.sin(3 * torch.pi * (pf2[:, 1] - pf2[:, 0]))) ** 2
        pf2 = pf2[torch.abs(cc) <= 2e-3]

        # 创建y
        x1 = torch.linspace(0, 1, 500, device=device, dtype=dtype).reshape(-1, 1)
        mask1 = (x1 < 0.1891).squeeze()
        mask2 = (x1 >= 0.1891).squeeze()
        y21 = 1 - 4 * x1[mask1]
        y22 = 0.3 - 0.3 * x1[mask2]
        y = torch.cat([x1, torch.cat([y21, y22], dim=0)], dim=1)

        all_R = []

        for i in range(len(H)):
            t = i / self.nt
            G = torch.sin(0.5 * torch.pi * torch.tensor(t, device=device, dtype=dtype))

            P1 = x1 + 0.25 * G * torch.sin(torch.pi * x1)
            P2 = 1 - x1 + 0.25 * G * torch.sin(torch.pi * x1)

            c1 = (4 * P1 + P2 - 1) * (0.3 * P1 + P2 - 0.3) < 0
            c2 = (P1 + P2 - 1.3) * (1.85 - P2 - P1 - \
                                    (0.3 * torch.sin(3 * torch.pi * (P2 - P1))) ** 2) > 0

            pf_data = torch.cat([P1, P2], dim=1)
            mask = ~(c1.squeeze() | c2.squeeze())
            pf_data = pf_data[mask]

            # 合并y
            pf_data = torch.cat([pf_data, y], dim=0)

            # 非支配排序（最大化）
            rank = non_dominate_rank(-pf_data)
            pf_data = pf_data[rank == 0]

            # 设置最小值为0
            min_idx1 = torch.argmin(pf_data[:, 0])
            pf_data[min_idx1, 0] = 0
            min_idx2 = torch.argmin(pf_data[:, 1])
            pf_data[min_idx2, 1] = 0

            # 合并pf2
            pf_data = torch.cat([pf_data, pf2], dim=0)

            # 非支配排序（最小化）
            rank = non_dominate_rank(pf_data)
            pf_data = pf_data[rank == 0]

            all_R.append(pf_data)

        return all_R

class DCP3(DCP):
    def __init__(self, d: int = 10, m: int = 2, ref_num: int = 1000, taut: int = 20, nt: int=10, maxG: int = 100):
        super().__init__(d, m, ref_num, taut, nt, maxG)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        计算dcp3问题的目标值和约束值
        X: (pop_size, n_var) tensor of decision variables
        """
        # 动态参数 t 的计算
        t = torch.floor(torch.tensor(self.Dtime / self.taut)) / self.nt

        # === 目标函数计算 (CalObj) ===
        G = torch.abs(torch.sin(0.5 * torch.pi * t))
        # g = 1+sum( sqrt(abs(PopDec(:,2:end)-G)) , 2 );
        g = 1 + torch.sum(torch.sqrt(torch.abs(X[:, 1:] - G)), dim=1, keepdim=True)

        # PopObj(:,1) =g.*PopDec(:,1)+G.^2;
        # PopObj(:,2) =g.*(1-PopDec(:,1))+G.^2;
        obj1 = g * X[:, 0:1] + G ** 2
        obj2 = g * (1 - X[:, 0:1]) + G ** 2

        PopObj = torch.cat([obj1, obj2], dim=1)
        PopObj[PopObj < 1e-18] = 0.0

        # === 约束计算 (Constraint) ===
        obj1, obj2 = PopObj[:, 0:1], PopObj[:, 1:2]

        # PopCon(:,1)=-((PopObj(:,1)-1).^2+(PopObj(:,2)-0.2).^2-0.3.^2).*((PopObj(:,1)-0.2).^2+(PopObj(:,2)-1).^2-0.3.^2);
        con1 = -((obj1 - 1) ** 2 + (obj2 - 0.2) ** 2 - 0.3 ** 2) * \
               ((obj1 - 0.2) ** 2 + (obj2 - 1) ** 2 - 0.3 ** 2)

        # PopCon(:,2)=PopObj(:,1).^2+PopObj(:,2).^2-4.^2;
        con2 = obj1 ** 2 + obj2 ** 2 - 4.0 ** 2

        # PopCon(:,3)=-(PopObj(:,1).^2+PopObj(:,2).^2-(3.1+0.2.*sin(4.*atan(PopObj(:,2)./PopObj(:,1))).^2).^2).*(PopObj(:,1).^2+PopObj(:,2).^2-(2.3).^2);
        # 使用 atan2 提高数值稳定性
        term_atan = torch.atan2(obj2, obj1)
        term_sin = torch.sin(4. * term_atan)
        term_inner_radius = (3.1 + 0.2 * term_sin ** 2) ** 2
        obj_radius_sq = obj1 ** 2 + obj2 ** 2

        con3 = -(obj_radius_sq - term_inner_radius) * (obj_radius_sq - 2.3 ** 2)

        PopCon = torch.cat([con1, con2, con3], dim=1)

        return PopObj, PopCon

    def pf(self) -> list:
        device = self.device
        dtype = torch.float32
        fe_values = torch.arange(0, self.maxG + 1, device=device, dtype=dtype)
        tt = torch.floor(fe_values /self.taut) / self.nt
        H = torch.sin(0.01 * torch.pi * tt)
        H = torch.unique(torch.round(H * 1e6) / 1e6)
        x1 = torch.linspace(0, 1, 500, device=device, dtype=dtype).reshape(-1, 1)
        x2 = torch.arange(0, 2.001, 0.001, device=device, dtype=dtype)
        n_x2 = len(x2)
        pf2_col1 = x2.repeat(n_x2)
        pf2_col2 = x2.repeat_interleave(n_x2)
        pf2 = torch.stack([pf2_col1, pf2_col2], dim=1)

        cc = ((pf2[:, 0] - 1) ** 2 + (pf2[:, 1] - 0.2) ** 2 - 0.3 ** 2) * \
             ((pf2[:, 1] - 1) ** 2 + (pf2[:, 0] - 0.2) ** 2 - 0.3 ** 2)
        pf2 = pf2[torch.abs(cc) <= 1e-3]
        pf2 = pf2[~((pf2[:, 0] < 1) & (pf2[:, 1] < 1))]
        all_R = []
        for i in range(len(H)):
            t = i / self.nt
            G = torch.abs(torch.sin(0.5 * torch.pi * torch.tensor(t, device=device, dtype=dtype)))
            G_sq = G ** 2
            P1 = x1 + G_sq
            P2 = 1 - x1 + G_sq
            pf_data = torch.cat([P1, P2], dim=1)
            c1 = ((pf_data[:, 0] - 1) ** 2 + (pf_data[:, 1] - 0.2) ** 2 - 0.3 ** 2) * \
                 ((pf_data[:, 1] - 1) ** 2 + (pf_data[:, 0] - 0.2) ** 2 - 0.3 ** 2) < 0
            pf_data = pf_data[~c1.squeeze()]
            if pf_data.shape[0] < len(x1):
                pf22 = pf2.clone()
                pf22 = pf22[~((pf22[:, 0] < G_sq) | (pf22[:, 1] < G_sq))]
                De = torch.max(pf_data[:, 0])
                pf_data = torch.cat([pf_data, pf22], dim=0)

                if 2 * De >= 2:
                    rank = non_dominate_rank(-pf_data)
                    pf_data = pf_data[rank == 0]

                rank = non_dominate_rank(pf_data)
                pf_data = pf_data[rank == 0]

            all_R.append(pf_data)

        return all_R

class DCP4(DCP):
    def __init__(self, d: int = 10, m: int = 2, ref_num: int = 1000, taut: int = 20, nt: int=10, maxG: int = 100):
        super().__init__(d, m, ref_num, taut, nt, maxG)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        t = torch.floor(torch.tensor(self.Dtime / self.taut)) / self.nt
        G = 2 * torch.floor(10 * torch.abs(torch.fmod(t + 1, 2) - 1) + 1e-4)

        x_shifted = X[:, 1:] - 0.5
        g_term = (x_shifted ** 2 - torch.cos(torch.pi * x_shifted) + 1) ** 2
        g = 1 + torch.sum(g_term, dim=1, keepdim=True)

        common_term = 0.25 * torch.sin(torch.pi * X[:, 0:1])
        obj1 = g * (X[:, 0:1] + common_term)
        obj2 = g * (1 - X[:, 0:1] + common_term)

        PopObj = torch.cat([obj1, obj2], dim=1)

        obj1_sq = PopObj[:, 0:1] ** 2
        obj2_sq = PopObj[:, 1:2] ** 2
        obj_sq_sum = obj1_sq + obj2_sq

        atan_term = torch.atan(PopObj[:, 1:2] / PopObj[:, 0:1])

        c11 = obj_sq_sum - (1.5 + 0.4 * torch.sin(4 * atan_term) ** 16) ** 2
        c12 = obj_sq_sum - (1.3 - 0.45 * torch.sin(G * atan_term) ** 2) ** 2

        PopCon = -c11 * c12

        return PopObj, PopCon

    def pf(self) -> list:
        device = self.device
        dtype = torch.float32

        # 计算时间步和H值
        fe_values = torch.arange(0, self.maxG + 1, device=device, dtype=dtype)
        tt = torch.floor(fe_values /self.taut) / self.nt
        H = torch.sin(0.01 * torch.pi * tt)
        H = torch.unique(torch.round(H * 1e6) / 1e6)  # 去掉重复的H值

        # 创建pf1
        x1 = torch.linspace(0, 1, 500, device=device, dtype=dtype).reshape(-1, 1)
        pf1_col1 = x1 + 0.25 * torch.sin(torch.pi * x1)
        pf1_col2 = 1 - x1 + 0.25 * torch.sin(torch.pi * x1)
        pf1 = torch.cat([pf1_col1, pf1_col2], dim=1)

        all_R = []

        for i in range(len(H)):
            pf = pf1.clone()
            t = i / self.nt

            # 计算G
            mod_val = torch.fmod(torch.tensor(t + 1, device=device, dtype=dtype),
                                 torch.tensor(2.0, device=device, dtype=dtype))
            G = 2 * torch.floor(10 * torch.abs(mod_val - 1) + 1e-4)

            # 计算atan(pf[:,1]/pf[:,0])，使用atan2更稳定
            atan_pf = torch.atan2(pf[:, 1], pf[:, 0])

            # 计算约束条件c11
            radius_sq = pf[:, 0] ** 2 + pf[:, 1] ** 2
            threshold1_sq = (1.5 + 0.4 * torch.sin(4 * atan_pf) ** 16) ** 2
            c11 = radius_sq - threshold1_sq

            # 计算约束条件c12
            threshold2_sq = (1.3 - 0.45 * torch.sin(G * atan_pf) ** 2) ** 2
            c12 = radius_sq - threshold2_sq

            # 组合约束：c11 * c12 < 0
            c1 = c11 * c12 < 0
            pf = pf[~c1]

            all_R.append(pf)

        return all_R

class DCP5(DCP):
    def __init__(self, d: int = 10, m: int = 2, ref_num: int = 1000, taut: int = 20, nt: int=10, maxG: int = 100):
        super().__init__(d, m, ref_num, taut, nt, maxG)
        self.lb = torch.tensor([0.0] + [-1.0] * (self.d - 1))
        self.ub = torch.tensor([1.0] * self.d)
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        t = torch.floor(torch.tensor(self.Dtime / self.taut)) / self.nt
        G = 0.5 * torch.abs(torch.sin(0.5 * torch.pi * t))

        x_shifted = X[:, 1:] - 0.5
        g_term = (x_shifted ** 2 - torch.cos(torch.pi * x_shifted) + 1) ** 2
        g = 1 + torch.sum(g_term, dim=1, keepdim=True)

        obj1 = g * X[:, 0:1]
        obj2 = g * (1 - X[:, 0:1])

        PopObj = torch.cat([obj1, obj2], dim=1)

        obj1_sq = PopObj[:, 0:1] ** 2
        obj2_sq = PopObj[:, 1:2] ** 2

        con1 = -((0.2 + G) * obj1_sq + PopObj[:, 1:2] - 2) * \
               (0.7 * obj1_sq + PopObj[:, 1:2] - 2.5)

        con2 = -(obj1_sq + obj2_sq - (0.6 + G) ** 2)

        PopCon = torch.cat([con1, con2], dim=1)

        return PopObj, PopCon

    def pf(self) -> list:
        device = self.device
        dtype = torch.float32

        # 计算时间步和H值
        fe_values = torch.arange(0, self.maxG + 1, device=device, dtype=dtype)
        tt = torch.floor(fe_values /self.taut) / self.nt
        H = torch.sin(0.01 * torch.pi * tt)
        H = torch.unique(torch.round(H * 1e6) / 1e6)  # 去掉重复的H值

        # 创建pf1
        pf1_col1 = torch.linspace(0, 1, 500, device=device, dtype=dtype).reshape(-1, 1)
        pf1_col2 = 1 - pf1_col1
        pf1 = torch.cat([pf1_col1, pf1_col2], dim=1)

        # 生成均匀分布的点（对应MATLAB的UniformPoint函数）
        # 使用均匀采样并归一化到单位球面
        X = self.sample
        X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True))

        all_R = []

        for i in range(len(H)):
            pf = pf1.clone()
            t = i / self.nt
            G = 0.5 * torch.abs(torch.sin(0.5 * torch.pi * torch.tensor(t, device=device, dtype=dtype)))

            # 应用约束条件 c1
            c1 = pf[:, 0] ** 2 + pf[:, 1] ** 2 - (0.6 + G) ** 2 < 0
            pf = pf[~c1]

            # 添加缩放后的X点
            scaled_X = (0.6 + G) * X
            pf = torch.cat([pf, scaled_X], dim=0)

            # 非支配排序，保留第一前沿（rank=0）
            rank = non_dominate_rank(-pf)
            pf = pf[rank == 0]

            all_R.append(pf)

        return all_R

class DCP6(DCP):
    def __init__(self, d: int = 10, m: int = 2, ref_num: int = 1000, taut: int = 20, nt: int=10, maxG: int = 100):
        super().__init__(d, m, ref_num, taut, nt, maxG)
        self.lb = torch.tensor([0.0] + [-1.0] * (self.d - 1))
        self.ub = torch.tensor([1.0] * self.d)
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        t = torch.floor(torch.tensor(self.Dtime / self.taut)) / self.nt
        G = torch.abs(torch.sin(0.5 * torch.pi * t)) ** 0.5

        g = 1 + 6 * torch.sum(X[:, 1:] ** 2, dim=1, keepdim=True)

        obj1 = g * X[:, 0:1]
        obj2 = g * (1 - X[:, 0:1])

        PopObj = torch.cat([obj1, obj2], dim=1)

        obj_sum = PopObj[:, 0:1] + PopObj[:, 1:2]
        obj_diff = PopObj[:, 1:2] - PopObj[:, 0:1]

        con1 = obj_sum - (4.5 + 0.08 * torch.sin(2 * torch.pi * (obj_diff / 1.6)))

        c21 = obj_sum - (2 - 0.08 * torch.sin(2 * torch.pi * (obj_diff / 1.5)))
        c22 = obj_sum - (3.2 - G - 0.08 * torch.sin(2 * torch.pi * (obj_diff / 1.5)))
        con2 = -c21 * c22

        cos_val = torch.cos(torch.tensor(-0.25 * torch.pi))
        sin_val = torch.sin(torch.tensor(-0.25 * torch.pi))

        rotated_x = PopObj[:, 0:1] * cos_val - PopObj[:, 1:2] * sin_val
        rotated_y = PopObj[:, 0:1] * sin_val + PopObj[:, 1:2] * cos_val

        con3 = -(rotated_x ** 2 / 1.1 ** 2 + rotated_y ** 2 / (0.1 + G) ** 2 - (0.1 + G) ** 2)

        PopCon = torch.cat([con1, con2, con3], dim=1)

        return PopObj, PopCon

    def _solve_ellipse_intersection(self, G, cos_angle, sin_angle, device, dtype):
        """
        求解椭圆与直线y=1-x的交点
        椭圆方程：(x*cos_angle - y*sin_angle)^2 / 1.1^2 + (x*sin_angle + y*cos_angle)^2 / (0.1+G)^2 - (0.1+G)^2 = 0
        直线方程：y = 1 - x
        """
        # 代入y = 1-x，求解x
        # 使用数值搜索找到最小的x值
        x_test = torch.linspace(0, 1, 1000, device=device, dtype=dtype)
        y_test = 1 - x_test

        rotated_x = x_test * cos_angle - y_test * sin_angle
        rotated_y = x_test * sin_angle + y_test * cos_angle

        ellipse_eq = (rotated_x ** 2) / (1.1 ** 2) + (rotated_y ** 2) / ((0.1 + G) ** 2) - (0.1 + G) ** 2

        # 找到椭圆上的点（接近0的点）
        mask = torch.abs(ellipse_eq) < 1e-3
        if mask.any():
            valid_x = x_test[mask]
            valid_y = y_test[mask]
            LongX = torch.min(valid_x)
            LongY = torch.min(valid_y)
        else:
            LongX = torch.tensor(0.0, device=device, dtype=dtype)
            LongY = torch.tensor(0.0, device=device, dtype=dtype)

        return LongX, LongY
    def pf(self) -> list:
        device = self.device
        dtype = torch.float32

        # 计算时间步和H值
        fe_values = torch.arange(0, self.maxG + 1, device=device, dtype=dtype)
        tt = torch.floor(fe_values /self.taut) / self.nt
        H = torch.sin(0.01 * torch.pi * tt)
        H = torch.unique(torch.round(H * 1e6) / 1e6)  # 去掉重复的H值

        # 创建pf1
        x1 = torch.linspace(0, 1, 500, device=device, dtype=dtype).reshape(-1, 1)
        pf1_col1 = x1
        pf1_col2 = 1 - x1
        pf1 = torch.cat([pf1_col1, pf1_col2], dim=1)

        # 创建y2（网格）
        x2 = torch.arange(0, 1.501, 0.001, device=device, dtype=dtype)
        n_x2 = len(x2)  # 1501
        y2_col1 = x2.repeat(n_x2)
        y2_col2 = x2.repeat_interleave(n_x2)
        y2 = torch.stack([y2_col1, y2_col2], dim=1)

        # 旋转角度和三角函数值
        angle =  torch.tensor(-0.25 * torch.pi, device=device, dtype=dtype)
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)

        all_R = []

        for i in range(len(H)):
            if i == 0:
                all_R.append(pf1.clone())
                continue
            pf = pf1.clone()
            pf2 = y2.clone()

            t = i / self.nt
            G = torch.abs(torch.sin(0.5 * torch.pi * torch.tensor(t, device=device, dtype=dtype))) ** 0.5

            # 对pf应用旋转和椭圆约束c1
            pf_rotated_x = pf[:, 0] * cos_angle - pf[:, 1] * sin_angle
            pf_rotated_y = pf[:, 0] * sin_angle + pf[:, 1] * cos_angle
            c1 = (pf_rotated_x ** 2) / (1.1 ** 2) + (pf_rotated_y ** 2) / ((0.1 + G) ** 2) - (0.1 + G) ** 2 < 0
            pf = pf[~c1]

            # 对pf2应用旋转和椭圆约束c2
            pf2_rotated_x = pf2[:, 0] * cos_angle - pf2[:, 1] * sin_angle
            pf2_rotated_y = pf2[:, 0] * sin_angle + pf2[:, 1] * cos_angle
            c2 = (pf2_rotated_x ** 2) / (1.1 ** 2) + (pf2_rotated_y ** 2) / ((0.1 + G) ** 2) - (0.1 + G) ** 2
            pf2 = pf2[~((c2 < -1e-4) | (c2 > 1e-2))]

            # 求解椭圆与y=1-x的交点（数值方法）
            LongX, LongY = self._solve_ellipse_intersection(G, cos_angle, sin_angle, device, dtype)

            # 过滤pf2
            pf2 = pf2[~((pf2[:, 0] < LongX) | (pf2[:, 1] < LongY))]

            # 合并pf和pf2
            pf_data = torch.cat([pf, pf2], dim=0)

            # 非支配排序
            rank = non_dominate_rank(pf_data)
            pf_data = pf_data[rank == 0]

            # 特殊情况：G == 1
            if torch.abs(G - 1.0) < 1e-6:
                X = self.sample
                X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True))
                pf_data = (0.1 + G) ** 2 * X

            all_R.append(pf_data)

        return all_R

class DCP7(DCP):
    def __init__(self, d: int = 10, m: int = 2, ref_num: int = 1000, taut: int = 20, nt: int=10, maxG: int = 100):
        super().__init__(d, m, ref_num, taut, nt, maxG)
        self.lb = torch.tensor([0.0] + [-1.0] * (self.d - 1))
        self.ub = torch.tensor([1.0] * self.d)
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        t = torch.floor(torch.tensor(self.Dtime / self.taut)) / self.nt
        G = torch.sin(0.5 * torch.pi * t)
        g = 1 + torch.sum((X[:, 1:] - G) ** 2, dim=1, keepdim=True)
        A = 0.02 * torch.sin((10 - torch.abs(torch.floor(10 * G))) * torch.pi * X[:, 0:1])
        obj1 = g * (X[:, 0:1] + A)
        obj2 = g * (1 - X[:, 0:1] + A)
        PopObj = torch.cat([obj1, obj2], dim=1)
        PopObj[torch.abs(PopObj) < 1e-15] = 0.0
        PopObj = PopObj + t
        obj_sum = PopObj[:, 0:1] + PopObj[:, 1:2]
        obj_diff = PopObj[:, 0:1] - PopObj[:, 1:2]
        con1 = -(obj_sum - torch.sin(5 * torch.pi * (obj_diff + 1)) ** 2 - G)
        PopCon = con1

        return PopObj, PopCon

    def pf(self) -> list:
        device = self.device
        dtype = torch.float32

        # 计算时间步和H值
        fe_values = torch.arange(0, self.maxG + 1, device=device, dtype=dtype)
        tt = torch.floor(fe_values /self.taut) / self.nt
        H = torch.sin(0.01 * torch.pi * tt)
        H = torch.unique(torch.round(H * 1e6) / 1e6)  # 去掉重复的H值

        # 创建x1（501个点）
        x1 = torch.linspace(0, 1, 501, device=device, dtype=dtype).reshape(-1, 1)

        all_R = []

        for i in range(len(H)):
            t = i / self.nt
            t_tensor = torch.tensor(t, device=device, dtype=dtype)

            # 计算G
            G = torch.sin(0.5 * torch.pi * t_tensor)

            # 计算A = 0.02*sin((10-abs(floor(10*G)))*pi*x1)
            floor_10G = torch.floor(10 * G)
            A = 0.02 * torch.sin((10 - torch.abs(floor_10G)) * torch.pi * x1)

            # 构造pf
            pf_col1 = x1 + A
            pf_col2 = 1 - x1 + A
            pf = torch.cat([pf_col1, pf_col2], dim=1)

            # 计算约束条件c1
            c1 = (pf[:, 0] + pf[:, 1] -
                  torch.sin(5 * torch.pi * (pf[:, 0] - pf[:, 1] + 1)) ** 2 - G)

            # 过滤c1<0的点
            pf = pf[c1 >= 0]

            # 最后加上t（对所有元素）
            pf_data = pf + t_tensor

            all_R.append(pf_data)

        return all_R


class DCP8(DCP):
    def __init__(self, d: int = 10, m: int = 2, ref_num: int = 1000, taut: int = 20, nt: int=10, maxG: int = 100):
        super().__init__(d, m, ref_num, taut, nt, maxG)
        self.lb = torch.tensor([0.0] + [-1.0] * (self.d - 1))
        self.ub = torch.tensor([1.0] * self.d)
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        t = torch.floor(torch.tensor(self.Dtime / self.taut)) / self.nt
        G = torch.sin(0.5 * torch.pi * t)

        y = X[:, 1:] - G
        g_term = (torch.abs(G) * y ** 2 - torch.cos(torch.pi * y) + 1) ** 2
        g = 1 + torch.sum(g_term, dim=1, keepdim=True)

        obj1 = g * X[:, 0:1]
        obj2 = g * (1 - X[:, 0:1])

        PopObj = torch.cat([obj1, obj2], dim=1)

        c11 = PopObj[:, 0:1] ** 1.5 + PopObj[:, 1:2] ** 1.5 - 1.2 ** 1.5
        c12 = PopObj[:, 0:1] ** 0.5 + PopObj[:, 1:2] ** 0.5 - (0.95 + 0.5 * torch.abs(G))

        obj_diff = PopObj[:, 1:2] - PopObj[:, 0:1]
        sin_term = 0.08 * torch.sin(2 * torch.pi * obj_diff)

        c21 = 0.8 * PopObj[:, 0:1] + PopObj[:, 1:2] - (2.5 + sin_term)
        c22 = (0.93 + torch.abs(G) / 3) * PopObj[:, 0:1] + PopObj[:, 1:2] - \
              (2.7 + torch.abs(G) / 2 + sin_term)

        con1 = -c11 * c12
        con2 = -c21 * c22

        PopCon = torch.cat([con1, con2], dim=1)

        return PopObj, PopCon

    def pf(self) -> list:
        device = self.device
        dtype = torch.float32

        # 计算时间步和H值
        fe_values = torch.arange(0, self.maxG + 1, device=device, dtype=dtype)
        tt = torch.floor(fe_values /self.taut) / self.nt
        H = torch.sin(0.01 * torch.pi * tt)
        H = torch.unique(torch.round(H * 1e6) / 1e6)  # 去掉重复的H值

        # 创建x1（501个点）
        x1 = torch.linspace(0, 1, 501, device=device, dtype=dtype).reshape(-1, 1)

        # 创建pf1：使用UniformPoint生成并归一化
        X = self.sample
        # pf1 = 1.2 * X / (sum(X^1.5))^(2/3)
        sum_X_pow = torch.sum(X ** 1.5, dim=1, keepdim=True) ** (2.0 / 3.0)
        pf1 = 1.2 * X / sum_X_pow

        all_R = []

        for i in range(len(H)):
            t = i / self.nt
            t_tensor = torch.tensor(t, device=device, dtype=dtype)

            # 计算G
            G = torch.sin(0.5 * torch.pi * t_tensor)

            # 构造pf：第一列x1，第二列1-x1
            pf_col1 = x1
            pf_col2 = 1 - x1
            pf = torch.cat([pf_col1, pf_col2], dim=1)

            # 计算约束条件c1
            c1 = (pf[:, 0] ** 0.5 + pf[:, 1] ** 0.5 -
                  (0.95 + 0.5 * torch.abs(G)))

            # 过滤c1>0的点（保留c1<=0的点）
            pf = pf[c1 <= 0]

            # 合并pf和pf1
            pf_data = torch.cat([pf, pf1], dim=0)

            # 非支配排序，只保留第一层
            rank = non_dominate_rank(pf_data)
            pf_data = pf_data[rank == 0]

            all_R.append(pf_data)

        return all_R

class DCP9(DCP):
    def __init__(self, d: int = 10, m: int = 2, ref_num: int = 1000, taut: int = 20, nt: int=10, maxG: int = 100):
        super().__init__(d, m, ref_num, taut, nt, maxG)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        t = torch.floor(torch.tensor(self.Dtime / self.taut)) / self.nt
        G = torch.abs(torch.sin(0.5 * torch.pi * t))

        r = int(torch.floor((self.d - 1) * G).item())
        indices = [i for i in range(self.d) if i != r]
        UnDec = X[:, indices]

        g = 1 + 10 * torch.sum((UnDec - G) ** 2, dim=1, keepdim=True)

        obj1 = g * X[:, r:r + 1]
        obj2 = g * (1 - X[:, r:r + 1])

        PopObj = torch.cat([obj1, obj2], dim=1)

        obj1_sq = PopObj[:, 0:1] ** 2
        obj2_sq = PopObj[:, 1:2] ** 2
        obj_sq_sum = obj1_sq + obj2_sq

        exp_term = 0.75 + 1.25 * G
        c11 = obj_sq_sum - (0.2 + G) ** 2
        c12 = PopObj[:, 0:1] ** exp_term + PopObj[:, 1:2] ** exp_term - 4. ** exp_term
        con1 = c11 * c12

        atan_term = torch.atan(PopObj[:, 1:2] / PopObj[:, 0:1])
        cos_term = torch.cos(6 * atan_term ** 3) ** 10

        term1 = (PopObj[:, 0:1] / (1 + 0.15 * cos_term)) ** 2
        term2 = (PopObj[:, 1:2] / (1 + 0.75 * cos_term)) ** 2

        c21 = obj_sq_sum - 1.6 ** 2
        c22 = 2.1 - term1 - term2
        con2 = c21 * c22

        PopCon = torch.cat([con1, con2], dim=1)

        return PopObj, PopCon

    def pf(self) -> list:
        device = self.device
        dtype = torch.float32

        # 计算时间步和H值
        fe_values = torch.arange(0, self.maxG + 1, device=device, dtype=dtype)
        tt = torch.floor(fe_values /self.taut) / self.nt
        H = torch.sin(0.01 * torch.pi * tt)
        H = torch.unique(torch.round(H * 1e6) / 1e6)  # 去掉重复的H值

        # 创建x1（501个点）
        x1 = torch.linspace(0, 1, 501, device=device, dtype=dtype).reshape(-1, 1)

        # 创建pf1：使用UniformPoint生成并归一化到单位向量
        X = self.sample
        # pf1 = X / sqrt(sum(X^2))
        X_norm = torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True))
        pf1 = X / X_norm

        all_R = []

        for i in range(len(H)):
            t = i / self.nt
            t_tensor = torch.tensor(t, device=device, dtype=dtype)

            # 计算G
            G = torch.sin(0.5 * torch.pi * t_tensor)
            G_abs = torch.abs(G)

            # 构造pf：第一列x1，第二列1-x1
            pf_col1 = x1
            pf_col2 = 1 - x1
            pf = torch.cat([pf_col1, pf_col2], dim=1)

            # 计算约束条件c1
            # c1 = (pf(:,1))^2 + (pf(:,2))^2 - ((0.2+abs(G)))^2
            c1 = pf[:, 0] ** 2 + pf[:, 1] ** 2 - (0.2 + G_abs) ** 2

            # 过滤c1<0的点（保留c1>=0的点）
            pf = pf[c1 >= 0]

            # 合并pf和缩放后的pf1
            pf1_scaled = (0.2 + G_abs) * pf1
            pf_data = torch.cat([pf, pf1_scaled], dim=0)

            # 非支配排序（注意：对-pf_data排序）
            rank = non_dominate_rank(-pf_data)
            pf_data = pf_data[rank == 0]

            all_R.append(pf_data)

        return all_R



