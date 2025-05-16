import argparse

from robot_vocalizer import RobotVocalizer
def main():
    # get api key from args 
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepgram-api-key", type=str, required=True)
    args = parser.parse_args()

    DEEPGRAM_API_KEY = args.deepgram_api_key
    vocalizer = RobotVocalizer(DEEPGRAM_API_KEY, "aura-2-apollo-en")
    vocalizer.vocalize("Spot is on a mission! Spot will go...")

if __name__ == "__main__":
    main()