import holidays
import pandas as pd


class HolidayUtil:
    def __init__(self, country="US"):
        try:
            holidays.country_holidays(country=country)
        except NotImplementedError:
            available_countries = (
                "https://github.com/dr-prodigy/python-holidays#available-countries"
            )
            error = "must be one of the available countries:\n%s" % available_countries
            raise ValueError(error)

        self.federal_holidays = getattr(holidays, country)(years=range(1950, 2100))

    def to_df(self):
        holidays_df = pd.DataFrame(
            sorted(self.federal_holidays.items()),
            columns=["holiday_date", "names"],
        )
        holidays_df.holiday_date = holidays_df.holiday_date.astype("datetime64")
        return holidays_df
