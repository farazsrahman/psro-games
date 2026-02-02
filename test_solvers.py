
import unittest
import numpy as np
import time
from solvers import solve_max_welfare_cce, solve_max_welfare_ce, fictitious_play_symmetric
from metrics import calculate_cce_regret, calculate_regret_ce
from games import sample_competitive_cooperative_interpolation_game

class TestSolvers(unittest.TestCase):
    
    def test_prisoners_dilemma(self):
        """
        PD Payoffs (T>R>P>S):
           C      D
        C  3,3    0,5
        D  5,0    1,1
        
        Symmetric U (Player 1):
           C  D
        C  3  0
        D  5  1
        
        Nash is (D, D). 
        CE set is {(D, D)} because D strictly dominates C.
        CCE set is also {(D, D)}.
        Max welfare should be 2.
        """
        U = np.array([[3, 0], 
                      [5, 1]])
        
        # CE
        mu_ce, w_ce = solve_max_welfare_ce(U)
        self.assertAlmostEqual(w_ce, 2.0, places=3)
        self.assertAlmostEqual(mu_ce[1,1], 1.0, places=3)
        
        # CCE
        w_cce, mu_cce = solve_max_welfare_cce(U)
        self.assertAlmostEqual(w_cce, 2.0, places=3)
        self.assertAlmostEqual(mu_cce[1,1], 1.0, places=3)
        
    def test_battle_of_sexes(self):
        """
        BoS:
           O      F
        O  3,2    0,0
        F  0,0    2,3
        
        U (P1):
           O  F
        O  3  0
        F  0  2
        
        U (P2 = U.T? No, symmetric function assumes U2=U.T, but BoS is usually not symmetric in that way)
        Wait, our solvers assume symmetric structure U2 = U.T
        If I pass U for P1, and solvers assume P2 has U^T, then:
        P1 plays row, P2 plays col.
        P1 payoffs: [[3, 0], [0, 2]]
        P2 payoffs: [[3, 0], [0, 2]]^T = [[3, 0], [0, 2]]
        
        Wait, U2 = U.T means P2(j, i) = U(i, j) ? No.
        It means P2's payoff matrix is the transpose of P1's?
        Usually symmetric game means P1(a, b) = P2(b, a).
        If input U is P1's payoff.
        P1(r, c) = U[r, c]
        P2(r, c) = U[c, r]  <-- This is what U.T implies typically for symmetric games construction
        
        So if U = [[3, 0], [0, 2]] (Coordination game)
        P1(O, O) = 3, P2(O, O) = 3 (from U[0,0])
        P1(F, F) = 2, P2(F, F) = 2 (from U[1,1])
        Off diagonals 0.
        
        This is a pure coordination game.
        Max Welfare: (O, O) -> 6.
        """
        U = np.array([[3, 0], 
                      [0, 2]])
                      
        mu_ce, w_ce = solve_max_welfare_ce(U)
        w_cce, mu_cce = solve_max_welfare_cce(U)
        
        self.assertAlmostEqual(w_ce, 6.0, places=3)
        self.assertAlmostEqual(w_cce, 6.0, places=3)
        self.assertAlmostEqual(mu_ce[0,0], 1.0, places=3)
    
    def test_chicken_game(self):
        """
        Chicken / Hawk-Dove
           C    D
        C  3,3  1,5
        D  5,1  0,0
        
        U (P1):
           C  D
        C  3  1
        D  5  0
        
        Symmetric.
        Nash: (C, D), (D, C) pure. Mixed (1/2, 1/2) -> exp 2.5?
        Max Welfare is (C, C) -> 6. 
        But (C, C) is not Nash. 
        Is (C, C) CCE?
        If everyone plays C. P1 dev C->D gets 5 > 3. So (C,C) is not CCE/CE.
        
        Correlated Eq can achieve better than worst Nash but (C, C) is unstable.
        The CCE/CE typically mixes between (C,D) and (D,C) and maybe (C,C).
        Max welfare CE tends to be a mix of (C,D), (D,C) and (C,C) such that dev constraints hold.
        Actually in Chicken, typical CE is 1/3, 1/3, 1/3 on (C,C), (C,D), (D,C).
        Payoffs: (3+3 + 1+5 + 5+1)/3 = (6+6+6)/3 = 6. Avg = 6.
        Let's see if our solver finds this.
        """
        U = np.array([[3, 1], 
                      [5, 0]])
        
        mu_ce, w_ce = solve_max_welfare_ce(U)
        w_cce, mu_cce = solve_max_welfare_cce(U)
        
        # print("Chicken CE:", mu_ce)
        # print("Chicken CCE:", mu_cce)
        
        self.assertGreaterEqual(w_cce, w_ce - 1e-5)
        
        # Check Regret
        reg_cce = calculate_cce_regret(U, mu_cce)
        reg_ce = calculate_regret_ce(U, mu_ce)
        
        self.assertLess(reg_cce, 1e-4, "CCE solution has high CCE regret")
        self.assertLess(reg_ce, 1e-4, "CE solution has high CE regret")
        
    def test_welfare_subset_property_random(self):
        """
        Verify that MaxWelfare(CCE) >= MaxWelfare(CE) on random games.
        Since CE is a subset of CCE.
        """
        rng = np.random.default_rng(123)
        for _ in range(5):
            game = sample_competitive_cooperative_interpolation_game(n_actions=10, alpha=0.5, rng=rng)
            U = game.payoffs_p1
            
            # Solve
            w_cce, mu_cce = solve_max_welfare_cce(U)
            mu_ce, w_ce = solve_max_welfare_ce(U)
            
            # Check Property
            self.assertGreaterEqual(w_cce, w_ce - 1e-4, 
                                    f"CCE Welfare {w_cce} should be >= CE Welfare {w_ce}")
            
            # Manual Check of Constraints
            P = mu_ce
            n = U.shape[0]
            max_viol = 0.0
            for i in range(n):
                for k in range(n):
                    if i == k: continue
                    # Constraint: sum_j P[i,j] * (U[i,j] - U[k,j]) >= 0
                    lhs = np.sum(P[i, :] * (U[i, :] - U[k, :]))
                    if lhs < -1e-6:
                        print(f"Constraint Viol P1 ({i}->{k}): {lhs:.6f}. P(rec {i})={np.sum(P[i,:]):.6f}")
                        max_viol = min(max_viol, lhs)

            print(f"Max Constraint Violation: {max_viol}")

            # Check Regrets
            reg_cce_on_cce = calculate_cce_regret(U, mu_cce)
            reg_ce_on_ce = calculate_regret_ce(U, mu_ce)
            print(f"Regret CE: {reg_ce_on_ce}")
            
            # If regret is high but constraint violation is low, it's the P(rec i) division.
            reg_cce_on_cce = calculate_cce_regret(U, mu_cce)
            reg_ce_on_ce = calculate_regret_ce(U, mu_ce)
            
            self.assertLess(reg_cce_on_cce, 1e-3, "CCE Regret on CCE solution failed")
            self.assertLess(reg_ce_on_ce, 1e-3, "CE Regret on CE solution failed")

    def test_profiling_solvers(self):
        print("\n--- Solver Profiling (Random 20x20 Game) ---")
        rng = np.random.default_rng(999)
        U = rng.uniform(-1, 1, size=(20, 20))
        
        start = time.time()
        solve_max_welfare_cce(U)
        cce_time = time.time() - start
        print(f"Max Welfare CCE (20 actions): {cce_time:.4f}s")
        
        start = time.time()
        solve_max_welfare_ce(U)
        ce_time = time.time() - start
        print(f"Max Welfare CE (20 actions): {ce_time:.4f}s")
        
        # CE is typically slower due to O(N^2) constraints vs O(N)
        # But for N=20 both should be reasonable.

if __name__ == '__main__':
    unittest.main()
