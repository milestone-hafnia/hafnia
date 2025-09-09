from typing import Optional

from pydantic import BaseModel


class LicenseType(BaseModel):
    name: str
    abbreviation: str
    url: str


LicenseTypes = [
    LicenseType(
        name="Creative Commons: Attribution-NonCommercial-ShareAlike 2.0 Generic",
        abbreviation="CC BY-NC-SA 2.0",
        url="http://creativecommons.org/licenses/by-nc-sa/2.0/",
    ),
    LicenseType(
        name="Creative Commons: Attribution-NonCommercial 2.0 Generic",
        abbreviation="CC BY-NC 2.0",
        url="http://creativecommons.org/licenses/by-nc/2.0/",
    ),
    LicenseType(
        name="Creative Commons: Attribution-NonCommercial-NoDerivs 2.0 Generic",
        abbreviation="CC BY-NC-ND 2.0",
        url="http://creativecommons.org/licenses/by-nc-nd/2.0/",
    ),
    LicenseType(
        name="Creative Commons: Attribution 2.0 Generic",
        abbreviation="CC BY 2.0",
        url="http://creativecommons.org/licenses/by/2.0/",
    ),
    LicenseType(
        name="Creative Commons: Attribution-ShareAlike 2.0 Generic",
        abbreviation="CC BY-SA 2.0",
        url="http://creativecommons.org/licenses/by-sa/2.0/",
    ),
    LicenseType(
        name="Creative Commons: Attribution-NoDerivs 2.0 Generic",
        abbreviation="CC BY-ND 2.0",
        url="http://creativecommons.org/licenses/by-nd/2.0/",
    ),
    LicenseType(
        name="Flickr: No known copyright restrictions",
        abbreviation="Flickr",
        url="http://flickr.com/commons/usage/",
    ),
    LicenseType(
        name="United States Government Work",
        abbreviation="US Gov",
        url="http://www.usa.gov/copyright.shtml",
    ),
]


def get_license_by_url(url: str) -> Optional[LicenseType]:
    for license in LicenseTypes:
        # To handle http urls
        license_url = license.url.replace("http://", "https://")
        url_https = url.replace("http://", "https://")
        if license_url == url_https:
            return license
    raise ValueError(f"License with URL '{url}' not found.")


def get_license_by_abbreviation(abbreviation: str) -> Optional[LicenseType]:
    for license in LicenseTypes:
        if license.abbreviation == abbreviation:
            return license
    raise ValueError(f"License with abbreviation '{abbreviation}' not found.")
