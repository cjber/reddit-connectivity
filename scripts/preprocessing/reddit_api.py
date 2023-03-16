import jsonlines
from psaw import PushshiftAPI
from tqdm import tqdm

from src.common.utils import Paths

SUBREDDITS = [line.strip() for line in open(Paths.RAW / "REDDIT" / "subreddits.txt")]
SUBREDDITS.remove("unitedkingdom")
SUBREDDITS.remove("Britain")


if __name__ == "__main__":
    api = PushshiftAPI(shards_down_behavior=None)

    for idx, subreddit in tqdm(enumerate(SUBREDDITS)):
        OUT_FILE = Paths.COMMENTS / f"{idx}_{subreddit}.jsonl"

        if not OUT_FILE.exists():
            with jsonlines.open(OUT_FILE, "a") as writer:
                print(f"Getting comments from {subreddit}.")
                gen = api.search_comments(subreddit=subreddit)
                for idx, obj in tqdm(enumerate(gen)):
                    out = {
                        "subreddit": obj.subreddit,
                        "text": obj.body,
                        "score": obj.score,
                        "thread": obj.link_id,
                        "created_utc": obj.created_utc,
                        "idx": idx,
                    }
                    try:
                        out["author"] = obj.author_fullname
                    except AttributeError:
                        out["author"] = "deleted"
                    writer.write(out)
