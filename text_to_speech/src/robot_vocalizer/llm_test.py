import argparse
import logging

from robot_vocalizer.plan_vocalizer import PlanVocalizer

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai-api-key", type=str, required=True)
    parser.add_argument("--deepgram-api-key", type=str, required=True)
    args = parser.parse_args()

    test_plan = """
    (goto-poi P1 O1)
    (inspect O1)
    (goto-poi P2 O2)
    (goto-poi P3 O3)
    (inspect O3)
    """

    plan_vocalizer = PlanVocalizer(args.openai_api_key, args.deepgram_api_key)
    plan_vocalizer.vocalize("Spot", test_plan)


if __name__ == "__main__":
    main()
