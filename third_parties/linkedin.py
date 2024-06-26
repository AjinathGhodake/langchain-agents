import os
import requests
from dotenv import load_dotenv

load_dotenv()


def scrape_linked_profile(linkedin_profile_url: str, mock: bool = False):
    if mock:
        linkedin_profile_url = linkedin_profile_url
        response = requests.get(
            linkedin_profile_url,
            timeout=10,
        )
    else:
        api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
        header_dic = {
            "Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'
        }
        response = requests.get(
            api_endpoint,
            params={"url": linkedin_profile_url},
            headers=header_dic,
            timeout=10,
        )
    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data


if __name__ == "__main__":
    print(
        scrape_linked_profile(
            linkedin_profile_url="www.linkedin.com/in/ajinath-ghodake",
            mock=True,
        )
    )
