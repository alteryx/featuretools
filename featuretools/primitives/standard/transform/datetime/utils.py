from typing import Optional, Tuple

import holidays
import pandas as pd


class HolidayUtil:
    def __init__(self, country="US"):
        try:
            country, subdivision = self.convert_to_subdivision(country)
            self.holidays = holidays.country_holidays(
                country=country,
                subdiv=subdivision,
            )
        except NotImplementedError:
            available_countries = (
                "https://github.com/dr-prodigy/python-holidays#available-countries"
            )
            error = "must be one of the available countries:\n%s" % available_countries
            raise ValueError(error)

        self.federal_holidays = getattr(holidays, country)(years=range(1950, 2075))

    def to_df(self):
        holidays_df = pd.DataFrame(
            sorted(self.federal_holidays.items()),
            columns=["holiday_date", "names"],
        )
        holidays_df.holiday_date = holidays_df.holiday_date.astype("datetime64[ns]")
        return holidays_df

    def convert_to_subdivision(self, country: str) -> Tuple[str, Optional[str]]:
        """Convert country to country + subdivision

           Created in response to library changes that changed countries to subdivisions

        Args:
            country (str): Original country name

        Returns:
            Tuple[str,Optional[str]]: country, subdivsion
        """
        return {
            "ENGLAND": ("GB", country),
            "NORTHERNIRELAND": ("GB", country),
            "PORTUGALEXT": ("PT", "Ext"),
            "PTE": ("PT", "Ext"),
            "SCOTLAND": ("GB", country),
            "UK": ("GB", country),
            "WALES": ("GB", country),
        }.get(country.upper(), (country, None))
