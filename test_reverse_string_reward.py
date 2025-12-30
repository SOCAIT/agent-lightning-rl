import sys
import os
from unittest.mock import MagicMock

# Mock openai before importing reverse_string_agent to avoid ImportError
# if the package is not installed in the current environment
sys.modules["openai"] = MagicMock()
sys.modules["agentlightning"] = MagicMock()
sys.modules["agentlightning.adapter"] = MagicMock()
sys.modules["agentlightning.litagent"] = MagicMock()
sys.modules["agentlightning.reward"] = MagicMock()
sys.modules["agentlightning.runner"] = MagicMock()
sys.modules["agentlightning.store"] = MagicMock()
sys.modules["agentlightning.tracer.agentops"] = MagicMock()
sys.modules["agentlightning.types"] = MagicMock()
sys.modules["rich.console"] = MagicMock()


# Add the current directory to sys.path to ensure imports work if run from project root
sys.path.append(os.path.join(os.getcwd(), 'agent-lightning-rl'))

try:
    from reverse_string_agent import parse_response_and_reward
except ImportError:
    # If running from inside agent-lightning-rl directory
    sys.path.append(os.getcwd())
    from reverse_string_agent import parse_response_and_reward

def test_parse_response_and_reward():
    test_cases = [
        {
            "name": "Perfect output",
            "input": "Hello",
            "output": "0lleH",
            "expected_reverse_reward": 1.0,
            "expected_replacement_reward": 1.0,
            "expected_total_reward": 1.0
        },
        {
            "name": "Correct reverse, no replacement",
            "input": "Hello",
            "output": "olleH",
            "expected_reverse_reward": 1.0,
            "expected_replacement_reward": 0.0,
            "expected_total_reward": 0.7
        },
        {
            "name": "Incorrect reverse",
            "input": "Hello",
            "output": "Hllo",
            "expected_reverse_reward": 0.0,
            "expected_replacement_reward": 0.0,
            "expected_total_reward": 0.0
        },
        {
            "name": "Complex with mixed case",
            "input": "Ioannis",
            "output": "s1nna01", # Corrected: sinnaoI -> s1nna01 (i->1, o->0, I->1)
            "expected_reverse_reward": 1.0,
            "expected_replacement_reward": 1.0,
            "expected_total_reward": 1.0
        },
        {
            "name": "Partial replacement",
            "input": "Oi", # Rev: iO -> Target: 10
            "output": "i0", # Correct reverse (iO), but replaced O->0, kept i as i. 
            # Reversal check: "i0" -> "io". "io" == "io". Rev Reward = 1.
            # Replacement check: Positions in "iO": 0 ('i'->1), 1 ('O'->0).
            # Output "i0": index 0 is 'i' (expected '1'), index 1 is '0' (expected '0').
            # Correct: 1/2 = 0.5
            "expected_reverse_reward": 1.0,
            "expected_replacement_reward": 0.5,
            "expected_total_reward": 0.7 * 1.0 + 0.3 * 0.5 # 0.85
        },
        {
            "name": "No o/i present",
            "input": "bcdf",
            "output": "fdcb",
            "expected_reverse_reward": 1.0,
            "expected_replacement_reward": 1.0, # Nothing to replace -> success
            "expected_total_reward": 1.0
        }
    ]

    print(f"{'Test Case':<30} | {'Rev':<5} | {'Rep':<5} | {'Tot':<5} | {'Result'}")
    print("-" * 75)

    all_passed = True
    for case in test_cases:
        total, rev, rep = parse_response_and_reward(case["input"], case["output"])
        
        # Check tolerances for float comparison
        pass_rev = abs(rev - case["expected_reverse_reward"]) < 1e-6
        pass_rep = abs(rep - case["expected_replacement_reward"]) < 1e-6
        pass_tot = abs(total - case["expected_total_reward"]) < 1e-6
        
        passed = pass_rev and pass_rep and pass_tot
        if not passed:
            all_passed = False
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{case['name']:<30} | {rev:<5.2f} | {rep:<5.2f} | {total:<5.2f} | {status}")
        
        if not passed:
             print(f"  Expected: Rev={case['expected_reverse_reward']}, Rep={case['expected_replacement_reward']}, Tot={case['expected_total_reward']}")
             print(f"  Got:      Rev={rev}, Rep={rep}, Tot={total}")

    if all_passed:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed.")

if __name__ == "__main__":
    test_parse_response_and_reward()
