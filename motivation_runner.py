import json
import uuid
import logging
from datetime import datetime
from daily_motivation import (
    get_access_token, get_account_info, get_project_people,
    get_quote, enhance_quote, quote_overlay_on_image,
    upload_image_to_basecamp, post_message
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # Load config
    with open("runner_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    session_id = cfg["session_id"]
    account_id = cfg["account_id"]
    project_id = cfg["project_id"]
    message_board_id = cfg["message_board_id"]
    cc_ids = set(str(i) for i in cfg.get("cc_ids", []))

    # Get access token
    access_token = get_access_token(session_id)
    if not access_token:
        logging.error("No access token available; exiting")
        return

    # Get people
    all_people = get_project_people(account_id, project_id, access_token)
    main_people = [p for p in all_people if str(p["id"]) not in cc_ids]
    cc_people = [p for p in all_people if str(p["id"]) in cc_ids]

    # Fetch quote and enhance
    quote, author = get_quote()
    enhanced = enhance_quote(quote, author)

    # Generate unique image
    base_image = "static/1.png"
    output_filename = f"quote_{uuid.uuid4().hex}.png"
    output_image = f"static/{output_filename}"
    quote_overlay_on_image(base_image, f"{quote} — {author}", output_path=output_image)

    # Upload to Basecamp
    attachable_sgid = upload_image_to_basecamp(
        account_id=account_id,
        access_token=access_token,
        image_path=output_image
    )

    # Post message
    success = post_message(
        account_id=account_id,
        project_id=project_id,
        message_board_id=message_board_id,
        access_token=access_token,
        quote=quote,
        author=author,
        project_people=main_people,
        cc_people=cc_people,
        mentions=None,
        test_mode=False,
        enhanced=enhanced,
        image_url=None,
        image_sgid=attachable_sgid
    )

    if success:
        logging.info("Motivational message posted successfully")
    else:
        logging.error("Failed to post motivational message")

if __name__ == "__main__":
    main()
