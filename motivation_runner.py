import json
import logging
import os
import sys
import random
import glob
from typing import Dict, List, Optional, Tuple

from daily_motivation import (
    load_access_token,
    get_project_people,
    get_quote,
    enhance_quote,
    quote_overlay_on_image,
    upload_image_to_basecamp,
    post_comment,
)

DEFAULT_CONFIG_PATH = "runner_config.json"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def read_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Required: "
            "account_id, project_id, parent_message_id, (optional: cc_ids)."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_people_by_cc(
    all_people: List[Dict], cc_ids: Optional[List[int]]
) -> Tuple[List[Dict], List[Dict]]:
    if not all_people:
        return [], []
    cc_id_str = {str(pid) for pid in (cc_ids or [])}
    cc_people = [p for p in all_people if str(p.get("id")) in cc_id_str]
    main_people = [p for p in all_people if str(p.get("id")) not in cc_id_str]
    return main_people, cc_people


def post_once(config: Dict) -> int:
    # only require these now
    required = ["account_id", "project_id", "parent_message_id"]
    for key in required:
        if key not in config:
            logging.error(f"Missing required config key: {key}")
            return 2

    # session_id comes from GitHub secret
    session_id: str = os.environ.get("SESSION_ID", "").strip()
    if not session_id:
        logging.error("SESSION_ID not found in environment variables")
        return 2

    account_id: int = int(config["account_id"])
    project_id: int = int(config["project_id"])
    parent_message_id: int = int(config["parent_message_id"])
    cc_ids: List[int] = list(config.get("cc_ids", []))

    token_data = load_access_token(session_id)
    if not token_data:
        logging.error("No valid access token found. Authenticate once locally and copy the token JSON to this server.")
        return 3

    access_token: str = token_data["access_token"]

    # Resolve mentions
    all_people = get_project_people(account_id, project_id, access_token)
    main_people, cc_people = split_people_by_cc(all_people, cc_ids)

    # Build content
    quote, author = get_quote()
    if not quote:
        logging.error("Failed to generate quote")
        return 4
    enhanced = enhance_quote(quote, author)

    # Choose random background
    candidates = glob.glob("static/*.png")
    base_image = random.choice(candidates) if candidates else "static/1.png"
    if not os.path.exists(base_image):
        base_image = "static/1.png"

    output_image = "output/img1.png"
    try:
        quote_overlay_on_image(base_image, f"{quote} — {author}", output_path=output_image)
    except Exception as e:
        logging.error(f"Failed to generate image: {e}")
        return 5

    attachable_sgid = upload_image_to_basecamp(
        account_id=account_id,
        access_token=access_token,
        image_path=output_image,
    )
    if not attachable_sgid:
        logging.error("Image upload failed; aborting post")
        return 6

    # Always post under the parent message
    ok = post_comment(
        account_id=account_id,
        project_id=project_id,
        parent_message_id=parent_message_id,
        access_token=access_token,
        image_sgid=attachable_sgid,
        quote=quote,
        author=author,
        project_people=main_people,
        cc_people=cc_people,
        enhanced=enhanced,
    )
    logging.info(f"Loaded config keys: {list(config.keys())}")
    logging.info(f"SESSION_ID present? {'yes' if session_id else 'no'}")


    if ok:
        logging.info("Post succeeded")
        return 0
    else:
        logging.error("Post failed")
        return 7


def main() -> int:
    configure_logging()
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    try:
        cfg = read_config(config_path)
    except Exception as e:
        logging.error(str(e))
        return 1

    return post_once(cfg)

if __name__ == "__main__":
    sys.exit(main())
