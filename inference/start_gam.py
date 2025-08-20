
from chat import chat
from utils.system_prompt import current_party, build_system_prompt

system_prompt = build_system_prompt(current_party)

def start_game():
    print("Singleplayer D&D — Critical Role edition.\n")
    while True:
        player_text = input("\nPlayer: ").strip()
        if player_text.lower() in ["quit", "exit"]:
            print("Game ended.")
            break
        ai_response = chat(player_text)
        if ai_response:
            print(ai_response)
        else:
            print("DM is silent... (no output from model)")

if __name__ == "__main__":
    start_game()
