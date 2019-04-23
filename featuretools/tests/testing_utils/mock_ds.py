# -*- coding: utf-8 -*-

import os
from datetime import datetime

import numpy as np
import pandas as pd

from featuretools import variable_types
from featuretools.entityset import EntitySet, Relationship
from featuretools.tests import integration_data


def make_ecommerce_files(with_integer_time_index=False, base_path=None, file_location='local',
                         split_by_time=False, compressed=False):
    """ Makes a entityset with the following shape:

          R         Regions
         / \\       .
        S   C       Stores, Customers
            |       .
            S   P   Sessions, Products
             \\ /   .
              L     Log
    """

    region_df = pd.DataFrame({'id': ['United States', 'Mexico'],
                              'language': ['en', 'sp']})

    store_df = pd.DataFrame({'id': range(6),
                             u'région_id': ['United States'] * 3 + ['Mexico'] * 2 + [np.nan],
                             'num_square_feet': list(range(30000, 60000, 6000)) + [np.nan]})

    product_df = pd.DataFrame({'id': ['Haribo sugar-free gummy bears', 'car',
                                      'toothpaste', 'brown bag', 'coke zero',
                                      'taco clock'],
                               'department': ["food", "electronics", "health",
                                              "food", "food", "electronics"],
                               'rating': [3.5, 4.0, 4.5, 1.5, 5.0, 5.0],
                               'url': ['google.com', 'https://www.featuretools.com/',
                                       'amazon.com', 'www.featuretools.com', 'bit.ly',
                                       'featuretools.com/demos/'],
                               })
    customer_times = {
        'signup_date': [datetime(2011, 4, 8), datetime(2011, 4, 9),
                        datetime(2011, 4, 6)],
        # some point after signup date
        'upgrade_date': [datetime(2011, 4, 10), datetime(2011, 4, 11),
                         datetime(2011, 4, 7)],
        'cancel_date': [datetime(2011, 6, 8), datetime(2011, 10, 9),
                        datetime(2012, 1, 6)],
        'date_of_birth': [datetime(1993, 3, 8), datetime(1926, 8, 2),
                          datetime(1993, 4, 20)]
    }
    if with_integer_time_index:
        customer_times['signup_date'] = [6, 7, 4]
        customer_times['upgrade_date'] = [18, 26, 5]
        customer_times['cancel_date'] = [27, 28, 29]
        customer_times['date_of_birth'] = [2, 1, 3]

    customer_df = pd.DataFrame({
        'id': [0, 1, 2],
        'age': [33, 25, 56],
        u'région_id': ['United States'] * 3,
        'cohort': [0, 1, 0],
        'cohort_name': ['Early Adopters', 'Late Adopters', 'Early Adopters'],
        'loves_ice_cream': [True, False, True],
        'favorite_quote': ['The proletariat have nothing to lose but their chains',
                           'Capitalism deprives us all of self-determination',
                           'All members of the working classes must seize the '
                           'means of production.'],
        'signup_date': customer_times['signup_date'],
        # some point after signup date
        'upgrade_date': customer_times['upgrade_date'],
        'cancel_date': customer_times['cancel_date'],
        'cancel_reason': ["reason_1", "reason_2", "reason_1"],
        'engagement_level': [1, 3, 2],
        'email': ['john.smith@example.com', '', 'team@featuretools.com'],
        'phone_number': ['5555555555', '555-555-5555', '1-(555)-555-5555'],
        'date_of_birth': customer_times['date_of_birth'],
    })

    ips = ['192.168.0.1', '2001:4860:4860::8888', '0.0.0.0',
           '192.168.1.1:2869', np.nan, '']
    filepaths = ['/home/user/docs/Letter.txt', './inthisdir', 'C:\\user\\docs\\Letter.txt',
                 '~/.rcinfo', '../../greatgrandparent', 'data.json']

    session_df = pd.DataFrame({'id': [0, 1, 2, 3, 4, 5],
                               'customer_id': [0, 0, 0, 1, 1, 2],
                               'device_type': [0, 1, 1, 0, 0, 1],
                               'device_name': ['PC', 'Mobile', 'Mobile', 'PC',
                                               'PC', 'Mobile'],
                               'ip': ips,
                               'filepath': filepaths, })

    times = list([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)] +
                 [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)] +
                 [datetime(2011, 4, 9, 10, 40, 0)] +
                 [datetime(2011, 4, 10, 10, 40, i) for i in range(2)] +
                 [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)] +
                 [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)])
    if with_integer_time_index:
        times = list(range(8, 18)) + list(range(19, 26))

    values = list([i * 5 for i in range(5)] +
                  [i * 1 for i in range(4)] +
                  [0] +
                  [i * 5 for i in range(2)] +
                  [i * 7 for i in range(3)] +
                  [np.nan] * 2)

    values_2 = list([i * 2 for i in range(5)] +
                    [i * 1 for i in range(4)] +
                    [0] +
                    [i * 2 for i in range(2)] +
                    [i * 3 for i in range(3)] +
                    [np.nan] * 2)

    values_many_nans = list([np.nan] * 5 +
                            [i * 1 for i in range(4)] +
                            [0] +
                            [np.nan] * 2 +
                            [i * 3 for i in range(3)] +
                            [np.nan] * 2)

    latlong = list([(values[i], values_2[i]) for i, _ in enumerate(values)])
    latlong2 = list([(values_2[i], -values[i]) for i, _ in enumerate(values)])
    zipcodes = list(['02116'] * 5 +
                    ['02116-3899'] * 4 +
                    ['0'] +
                    ['1234567890'] * 2 +
                    ['12345-6789'] * 2 +
                    [np.nan] +
                    [''] * 2)
    countrycodes = list(['US'] * 5 +
                        ['AL'] * 4 +
                        [np.nan] * 2 +
                        [''] * 3 +
                        ['ALB'] * 2 +
                        ['USA'])
    subregioncodes = list(['US-AZ'] * 5 +
                          ['US-MT'] * 4 +
                          [np.nan] * 2 +
                          [''] +
                          ['UG-219'] * 2 +
                          ['ZM-06'] * 3)
    log_df = pd.DataFrame({
        'id': range(17),
        'session_id': [0] * 5 + [1] * 4 + [2] * 1 + [3] * 2 + [4] * 3 + [5] * 2,
        'product_id': ['coke zero'] * 3 + ['car'] * 2 + ['toothpaste'] * 3 +
        ['brown bag'] * 2 + ['Haribo sugar-free gummy bears'] +
        ['coke zero'] * 4 + ['taco clock'] * 2,
        'datetime': times,
        'value': values,
        'value_2': values_2,
        'latlong': latlong,
        'latlong2': latlong2,
        'zipcode': zipcodes,
        'countrycode': countrycodes,
        'subregioncode': subregioncodes,
        'value_many_nans': values_many_nans,
        'priority_level': [0] * 2 + [1] * 5 + [0] * 6 + [2] * 2 + [1] * 2,
        'purchased': [True] * 11 + [False] * 4 + [True, False],
        'comments': [coke_zero_review()] + ['I loved it'] * 2 +
        car_reviews() + toothpaste_reviews() +
        brown_bag_reviews() + [gummy_review()] +
        ['I loved it'] * 4 + taco_clock_reviews()
    })
    filenames = {}
    for entity, df in [(u'régions', region_df),
                       ('stores', store_df),
                       ('products', product_df),
                       ('customers', customer_df),
                       ('sessions', session_df),
                       ('log', log_df)]:
        if split_by_time and entity == 'log':
            df1 = df.iloc[:10]
            df2 = df.iloc[10:]
            df = [df1, df2]
        filenames[entity] = save_to_csv(entity, df, base_path=base_path, file_location=file_location,
                                        with_integer_time_index=with_integer_time_index, compressed=compressed)
    return filenames


def data_dir(base_path=None):
    dirname = os.path.dirname(integration_data.__file__)
    if base_path is None:
        return os.path.abspath(dirname)
    return os.path.join(base_path, dirname)


def save_to_csv(entity_id, df, base_path=None, file_location='local',
                with_integer_time_index=False, compressed=False):
    path = entity_filename(entity_id, base_path, file_location=file_location,
                           with_integer_time_index=with_integer_time_index,
                           compressed=compressed)
    ext = '.csv'
    compression = None
    if compressed:
        ext = '.gzip'
        compression = ext[1:]
    if file_location == 's3':
        pass
        # src = StringIO()
        # df.to_csv(src, index=False,
        #         compression=compression,
        #         encoding='utf-8')
        # new_path = path.split('s3://')[1]
        # parts = new_path.split('/')
        # bucket = parts[0]
        # new_path = '/'.join(parts[1:])
        # src.seek(0)
        # upload_file(src, new_path, bucket)
    else:
        if isinstance(df, list):
            for i, d in enumerate(df):
                p = path.replace(ext, '-{}{}'.format(i, ext))
                d.to_csv(p, index=False,
                         compression=compression,
                         encoding='utf-8')
            path = path.replace(ext, '-*{}'.format(ext))
        else:
            df.to_csv(path, index=False,
                      compression=compression,
                      encoding='utf-8')
    return path


def entity_filename(entity, base_path=None, file_location='local',
                    with_integer_time_index=False, glob=False, compressed=False):
    if file_location == 's3':
        dirname = "s3://featuretools-static/test_ecommerce/"
    else:
        dirname = data_dir(base_path)

    ext = '.csv'
    if compressed:
        ext = '.gzip'

    if with_integer_time_index:
        entity = entity + "_int"
    if glob:
        path = os.path.join(dirname, entity + "-*" + ext)
    else:
        path = os.path.join(dirname, entity + ext)
    return path


def make_variable_types(with_integer_time_index=False):
    region_variable_types = {
        'id': variable_types.Categorical,
        'language': variable_types.Categorical
    }

    store_variable_types = {
        'id': variable_types.Categorical,
        u'région_id': variable_types.Id
    }

    product_variable_types = {
        'id': variable_types.Categorical,
        'rating': variable_types.Numeric,
        'department': variable_types.Categorical,
        'url': variable_types.URL,
    }

    customer_variable_types = {
        'id': variable_types.Categorical,
        'age': variable_types.Numeric,
        u'région_id': variable_types.Id,
        'loves_ice_cream': variable_types.Boolean,
        'favorite_quote': variable_types.Text,
        'signup_date': variable_types.Datetime,
        'upgrade_date': variable_types.Datetime,
        'cancel_date': variable_types.Datetime,
        'cancel_reason': variable_types.Categorical,
        'engagement_level': variable_types.Ordinal,
        'email': variable_types.EmailAddress,
        'phone_number': variable_types.PhoneNumber,
        'date_of_birth': variable_types.DateOfBirth,
    }

    session_variable_types = {
        'id': variable_types.Categorical,
        'customer_id': variable_types.Id,
        'device_type': variable_types.Categorical,
        'ip': variable_types.IPAddress,
        'filepath': variable_types.FilePath,
    }

    log_variable_types = {
        'id': variable_types.Categorical,
        'session_id': variable_types.Id,
        'product_id': variable_types.Categorical,
        'datetime': variable_types.Datetime,
        'value': variable_types.Numeric,
        'value_2': variable_types.Numeric,
        'latlong': variable_types.LatLong,
        'latlong2': variable_types.LatLong,
        'zipcode': variable_types.ZIPCode,
        'countrycode': variable_types.CountryCode,
        'subregioncode': variable_types.SubRegionCode,
        'value_many_nans': variable_types.Numeric,
        'priority_level': variable_types.Ordinal,
        'purchased': variable_types.Boolean,
        'comments': variable_types.Text
    }
    if with_integer_time_index:
        log_variable_types['datetime'] = variable_types.Numeric
        customer_variable_types['signup_date'] = variable_types.Numeric
        customer_variable_types['upgrade_date'] = variable_types.Numeric
        customer_variable_types['cancel_date'] = variable_types.Numeric
        customer_variable_types['date_of_birth'] = variable_types.Numeric

    return {
        'customers': customer_variable_types,
        'sessions': session_variable_types,
        'log': log_variable_types,
        'products': product_variable_types,
        'stores': store_variable_types,
        u'régions': region_variable_types
    }


def make_time_indexes(with_integer_time_index=False):
    return {'customers': {'name': 'signup_date',
                          'secondary': {
                              "cancel_date": ["cancel_reason"]}
                          },
            'log': {'name': 'datetime',
                    'secondary': None
                    }
            }


def latlong_unstringify(latlong):
    lat = float(latlong.split(", ")[0].replace("(", ""))
    lon = float(latlong.split(", ")[1].replace(")", ""))
    return (lat, lon)


def make_ecommerce_entityset(with_integer_time_index=False, base_path=None, save_files=True, file_location='local',
                             split_by_time=False, compressed=False, entityset_type=EntitySet):
    if file_location == 'local' and save_files:
        filenames = make_ecommerce_files(with_integer_time_index, base_path=base_path, file_location=file_location,
                                         split_by_time=split_by_time, compressed=compressed)
        entities = filenames.keys()
    else:
        entities = [u'régions', 'stores', 'products',
                    'customers', 'sessions', 'log']
        filenames = {e: entity_filename(e, base_path, file_location=file_location,
                                        glob=(split_by_time and e == 'log'),
                                        compressed=compressed)
                     for e in entities}
    id = 'ecommerce'
    if with_integer_time_index:
        id += "_int_time_index"
    if split_by_time:
        id += "_glob"

    variable_types = make_variable_types(
        with_integer_time_index=with_integer_time_index)
    time_indexes = make_time_indexes(
        with_integer_time_index=with_integer_time_index)

    es = entityset_type(id=id)

    for entity in entities:
        time_index = time_indexes.get(entity, None)
        ti_name = None
        secondary = None
        if time_index is not None:
            ti_name = time_index['name']
            secondary = time_index['secondary']
        df = pd.read_csv(filenames[entity], encoding='utf-8', engine='python')
        if entity == "customers":
            df["id"] = pd.Categorical(df['id'])
        if entity == 'sessions':
            # This should be changed back when converted to an EntitySet
            df['customer_id'] = pd.Categorical(df['customer_id'])
        if entity == 'log':
            df['latlong'] = df['latlong'].apply(latlong_unstringify)
            df['latlong2'] = df['latlong2'].apply(latlong_unstringify)

        es.entity_from_dataframe(entity,
                                 df,
                                 index='id',
                                 variable_types=variable_types[entity],
                                 time_index=ti_name,
                                 secondary_time_index=secondary)

    es.normalize_entity('customers', 'cohorts', 'cohort',
                        additional_variables=['cohort_name'],
                        make_time_index=True,
                        new_entity_time_index='cohort_end')

    es.add_relationships(
        [Relationship(es[u'régions']['id'], es['customers'][u'région_id']),
         Relationship(es[u'régions']['id'], es['stores'][u'région_id']),
         Relationship(es['customers']['id'], es['sessions']['customer_id']),
         Relationship(es['sessions']['id'], es['log']['session_id']),
         Relationship(es['products']['id'], es['log']['product_id'])])

    return es


def coke_zero_review():
    return u"""
When it comes to Coca-Cola products, people tend to be die-hard fans. Many of us know someone who can't go a day without a Diet Coke (or two or three). And while Diet Coke has been a leading sugar-free soft drink since it was first released in 1982, it came to light that young adult males shied away from this beverage — identifying diet cola as a woman's drink. The company's answer to that predicament came in 2005 - in the form of a shiny black can - with the release of Coca-Cola Zero.

While Diet Coke was created with its own flavor profile and not as a sugar-free version of the original, Coca-Cola Zero aims to taste just like the "real Coke flavor." Despite their polar opposite advertising campaigns, the contents and nutritional information of the two sugar-free colas is nearly identical. With that information in hand we at HuffPost Taste needed to know: Which of these two artificially-sweetened Coca-Cola beverages actually tastes better? And can you even tell the difference between them?

Before we get to the results of our taste test, here are the facts:


Diet Coke

Motto: Always Great Tast
Nutritional Information: Many say that a can of Diet Coke actually contains somewhere between 1-4 calories, but if a serving size contains fewer than 5 calories a company is not obligated to note it in its nutritional information. Diet Coke's nutritional information reads 0 Calories, 0g Fat, 40mg Sodium, 0g Total Carbs, 0g Protein.

Ingredients: Carbonated water, caramel color, aspartame, phosphoric acid, potassium benzonate, natural flavors, citric acid, caffeine.

Artificial sweetener: Aspartame


Coca-Cola Zero
Motto: Real Coca-Cola Taste AND Zero Calories

Nutritional Information: While the label clearly advertises this beverage as a zero calorie cola, we are not entirely certain that its minimal calorie content is simply not required to be noted in the nutritional information. Coca-Cola Zero's nutritional information reads 0 Calories, 0g Fat, 40mg Sodium, 0g Total Carbs, 0g Protein.

Artificial sweetener: Aspartame and acesulfame potassium

Ingredients: Carbonated water, caramel color, phosphoric acid, aspartame, potassium benzonate, natural flavors, potassium citrate, acesulfame potassium, caffeine.

The Verdict:
Twenty-four editors blind-tasted the two cokes, side by side, and...

54 percent of our tasters were able to distinguish Diet Coke from Coca-Cola Zero
50 percent of our tasters preferred Diet Coke to Coca-Cola Zero, and vice versa
Here’s what our tasters thought of the two sugar-free soft drinks:

Diet Coke: "Tastes fake right away." "Much fresher brighter, crisper." "Has the wonderful flavors of Diet Coke’s artificial sweeteners."

Coca-Cola Zero: "Has more of a sharply sweet aftertaste I associate with diet sodas." "Tastes more like regular coke, less like fake sweetener." "Has an odd taste." "Tastes more like regular." "Very sweet."

Overall comments: "That was a lot more difficult than I though it would be." "Both equally palatable." A few people said Diet Coke tasted much better ... unbeknownst to them, they were actually referring to Coca-Cola Zero.

IN SUMMARY: It is a real toss up. There is not one artificially-sweetened Coca-Cola beverage that outshines the other. So how do people choose between one or the other? It is either a matter of personal taste, or maybe the marketing campaigns will influence their choice.
"""


def gummy_review():
    return u"""
The place: BMO Harris Bradley Center
The event: Bucks VS Spurs
The snack: Satan's Diarrhea Hate Bears made by Haribo

I recently took my 4 year old son to his first NBA game. He was very excited to go to the game, and I was excited because we had fantastic seats. Row C center court to be exact. I've never sat that close before. I've never had to go DOWN stairs to get to my seats. 24 stairs to get to my seats to be exact.

His favorite candy is Skittles. Mine are anything gummy. I snuck in a bag of skittles for my son, and grabbed a handful of gummy bears for myself, to be later known as Satan's Diarrhea Hate Bears, that I received for Christmas in bulk from my parents, and put them in a zip lock bag.

After the excitement of the 1st quarter has ended I take my son out to get him a bottled water and myself a beer. We return to our seats to enjoy our candy and drinks.

..............fast forward until 1 minute before half time...........

I have begun to sweat a sweat that is only meant for a man on mile 19 of a marathon. I have kicked out my legs out so straight that I am violently pushing the gentleman wearing a suit seat in front of me forward. He is not happy, I do not care. My hands are on the side of my seat not unlike that of a gymnast on a pommel horse, lifting me off my chair. My son is oblivious to what is happening next to him, after all, there is a mascot running around somewhere and he is eating candy.

I realize that at some point in the very near to immediate future I am going to have to allow this lava from Satan to forcefully expel itself from my innards. I also realize that I have to walk up 24 stairs just to get to level ground in hopes to make it to the bathroom. I’ll just have to sit here stiff as a board for a few moments waiting for the pain to subside. About 30 seconds later there is a slight calm in the storm of the violent hurricane that is going on in my lower intestine. I muster the courage to gently relax every muscle in my lower half and stand up. My son stands up next to me and we start to ascend up the stairs. I take a very careful and calculated step up the first stair. Then a very loud horn sounds. Halftime. Great. It’s going to be crowded. The horn also seems to have awaken the Satan's Diarrhea Hate Bears that are having a mosh pit in my stomach. It literally felt like an avalanche went down my stomach and I again have to tighten every muscle and stand straight up and focus all my energy on my poor sphincter to tighten up and perform like it has never performed before. Taking another step would be the worst idea possible, the flood gates would open. Don’t worry, Daddy has a plan. I some how mumble the question, “want to play a game?” to my son, he of course says “yes”. My idea is to hop on both feet allllll the way up the stairs, using the center railing to propel me up each stair. My son is always up for a good hopping game, so he complies and joins in on the “fun”. Some old lady 4 steps up thinks its cute that we are doing this, obviously she wasn’t looking at the panic on my face. 3 rows behind her a man about the same age as me, who must have had similar situations, notices the fear/panic/desperation on my face understands the danger that I along with my pants and anyone within a 5 yard radius spray zone are in. He just mouths the words “good luck man” to me and I press on. Half way up and there is no leakage, but my legs are getting tired and my sphincter has never endured this amount of pressure for this long of time. 16 steps/hops later…….4 steps to go…….My son trips and falls on the stairs, I have two options: keep going knowing he will catch up or bend down to pick him up relieving my sphincter of all the pressure and commotion while ruining the day of roughly the 50 people that are now watching a grown man hop up stairs while sweating profusely next to a 4 year old boy.

Luckily he gets right back up and we make it to the top of the stairs. Good, the hard part was over. Or so I thought. I managed to waddle like a penguin, or someone who is about to poop their pants in 2.5 seconds, to the men's room only to find that every stall is being used. EVERY STALL. It's halftime, of course everyone has to poop at that moment. I don't know if I can wait any longer, do I go ahead and fulfil the dream of every high school boy and poop in the urinal? What kind of an example would that set for my son? On the other hand, what kind of an example would it be for his father to fill his pants with a substance that probably will be unrecognizable to man. Suddenly a stall door opens, and I think I manage to actually levitate over to the stall. I my son follows me in, luckily it was the handicap stall so there was room for him to be out of the way. I get my pants off and start to sit. I know what taking a giant poo feels like. I also know what vomiting feels like. I can now successfully say that I know what it is like to vomit out my butt. I wasn't pooping, those Satan's Diarrhea Hate Bears did something to my insides that made my sphincter vomit our the madness.

I am now conscious of my surroundings. Other than the war that the bottom half of my body is currently having with this porcelain chair, it is quiet as a pin drop in the bathroom. The other men in there can sense that something isn't right, no one has heard anyone ever poop vomit before.

I can sense that the worst part is over. But its not stopping, nor can I physically stop it at this point, I am leaking..it's horrible. I call out "does anyone have a diaper?" hoping that some gentleman was changing a baby. Nothing. No one said a word. I know people are in there, I can see the toes of shoes pointed in my direction under the stall.. "DOES ANYONE HAVE A DIAPER!?!" I am screaming, my son is now crying, he thinks he is witnessing the death of his father. I can't even assure him that I will make it.

Not a word was said, but a diaper was thrown over the stall. I catch it, line my underwear with it, put my pants back on, and walk out of that bathroom like a champ. We go straight to our seats, grab out coats and go home. As we are walking out, the gentleman that wished me good luck earlier simply put his fist out, and I happily bumped it.

My son asks me, "Daddy, why are we leaving early?"
"Well son, I need to change my diaper"
"""


def taco_clock_reviews():
    return [u"""
This timer does what it is supposed to do. Setup is elementary. Replacing the old one (after 12 years) was relatively easy. It has performed flawlessly since. I'm delighted I could find an esoteric product like this at Amazon. Their service, and the customer reviews, are just excellent.
""",
            """
Funny, cute clock. A little spendy for how light the clock is, but its hard to find a taco clock.
"""
            ]


def brown_bag_reviews():
    return [u"""
These bags looked exactly like I'd hoped, however, the handles broke off of almost every single bag as soon as items were placed in them! I used these as gift bags for out-of-town guests at my wedding, so imagine my embarassment as the handles broke off as I was handing them out. I would not recommend purchaing these bags unless you plan to fill them with nothing but paper! Anything heavier will cause the handles to snap right off.
""", u"""
I purchased these in August 2014 from Big Blue Supplies. I have no problem with the seller, these arrived new condition, fine shape.

I do have a slight problem with the bags. In case someone might want to know, the handles on these bags are set inside against the top. Then a piece of Kraft type packing tape is placed over the handles to hold them in place. On some of the bags, the tape is already starting to peel off. I would be really hesitant about using these bags unless I reinforced the current tape with a different adhesive.

I will keep the bags, and make a tape of a holiday or decorative theme and place over in order to make certain the handles stay in place.

Also in case anybody is wondering, the label on the plastic packaging bag states these are from ORIENTAL TRADING COMPANY. On the bottom of each bag it is stamped MADE IN CHINA. Again, I will be placing a sticker over that.

Even the dollar store bags I normally purchase do not have that stamped on the bottom in such prominent lettering. I purchased these because they were plain and I wanted to decorate them.

I do not think I would purchase again for all the reasons stated above.

Another thing for those still wanting to purchase, the ones I received were: 12 3/4 inches high not including handle, 10 1/4 inches wide and a 5 1/4 inch depth.
"""]


def car_reviews():
    return [u"""
The full-size pickup truck and the V-8 engine were supposed to be inseparable, like the internet and cat videos. You can’t have one without the other—or so we thought.

In America’s most popular vehicle, the Ford F-150, two turbocharged six-cylinder engines marketed under the EcoBoost name have dethroned the naturally aspirated V-8. Ford’s new 2.7-liter twin-turbo V-6 is the popular choice, while the 3.5-liter twin-turbo V-6 is the top performer. The larger six allows for greater hauling capacity, accelerates the truck more quickly, and swills less gas in EPA testing than the V-8 alternative. It’s enough to make even old-school truck buyers acknowledge that there actually is a replacement for displacement.

And yet a V-8 in a big pickup truck still feels so natural, so right. In the F-150, the Coyote 5.0-liter V-8 is tuned for torque more so than power, yet it still revs with an enthusiastic giddy-up that reminds us that this engine’s other job is powering the Mustang. The response follows the throttle pedal faithfully while the six-speed automatic clicks through gears smoothly and easily. Together they pull this 5220-pound F-150 to 60 mph in 6.3 seconds, which is 0.4 second quicker than the 5.3-liter Chevrolet Silverado with the six-speed automatic and 0.9 second quicker than the 5.3 Silverado with the new eight-speed auto. The 3.5-liter EcoBoost, though, can do the deed another half-second quicker, but its synthetic soundtrack doesn’t have the rich, multilayered tone of the V-8.

It wasn’t until we saddled our test truck with a 6400-pound trailer (well under its 9000-pound rating) that we fully understood the case for upgrading to the 3.5-liter EcoBoost. The twin-turbo engine offers an extra 2500 pounds of towing capability and handles lighter tasks with considerably less strain. The 5.0-liter truck needs more revs and a wider throttle opening to accelerate its load, so we were often coaxed into pressing the throttle to the floor for even modest acceleration. The torquier EcoBoost engine offers a heartier response at part throttle.

In real-world, non-towing situations, the twin-turbo 3.5-liter doesn’t deliver on its promise of increased fuel economy, with both the 5.0-liter V-8 and that V-6 returning 16 mpg in our hands. But given the 3.5-liter’s virtues, we can forgive it that trespass.

Trucks Are the New Luxury

Pickups once were working-class transportation. Today, they’re proxy luxury vehicles—or at least that’s how they’re priced. If you think our test truck’s $57,240 window sticker is steep, consider that our model, the Lariat, is merely a mid-spec trim. There are three additional grades—King Ranch, Platinum, and Limited—positioned and priced above it, plus the 3.5-liter EcoBoost that costs an extra $400 as well as a plethora of options to inflate the price past 60 grand. Squint and you can almost see the six-figure trucks of the future on the horizon.

For the most part, though, the equipment in this particular Lariat lives up to the price tag. The driver and passenger seats are heated and cooled, with 10-way power adjustability and supple leather. The technology includes blind-spot monitoring, navigation, and a 110-volt AC outlet. Nods to utility include spotlights built into the side mirrors and Ford’s Pro Trailer Backup Assist, which makes reversing with a trailer as easy as turning a tiny knob on the dashboard.

Middle-Child Syndrome

In the F-150, Ford has a trifecta of engines (the fourth, a naturally aspirated 3.5-liter V-6, is best left to the fleet operators). The 2.7-liter twin-turbo V-6 delivers remarkable performance at an affordable price. The 3.5-liter twin-turbo V-6 is the workhorse, with power, torque, and hauling capability to spare. Compared with those two logical options, the middle-child 5.0-liter V-8 is the right-brain choice. Its strongest selling points may be its silky power delivery and the familiar V-8 rumble. That’s a flimsy argument when it comes to rationalizing a $50,000-plus purchase, though, so perhaps it’s no surprise that today’s boosted six-cylinders are now the engines of choice in the F-150.
""",
            """
THE GOOD
The Tesla Model S 90D's electric drivetrain is substantially more efficient than any internal combustion engine, and gives the car smooth and quick acceleration. All-wheel drive comes courtesy of a smart dual motor system. The new Autopilot feature eases the stress of stop-and-go traffic and long road trips.

THE BAD
Even at Tesla's Supercharger stations, recharging the battery takes significantly longer than refilling an internal combustion engine car's gas tank, limiting where you can drive. Tesla hasn't improved its infotainment system much from the Model S' launch.

THE BOTTOM LINE
Among the different flavors of Tesla Model S, the 90D is the one to get, exhibiting the best range and all-wheel drive, while offering an uncomplicated, next-generation driving experience that shows very well against equally priced competitors.


REVIEW  SPECIFICATIONS  PHOTOS
Roadshow Automobiles Tesla 2016 Tesla Model S
Having tested driver assistance systems in many cars, and even ridden in fully self-driving cars, I should have been ready for Tesla's new Autopilot feature. But engaging it while cruising the freeway in the Model S 90D, I kept my foot hovering over the brake.

My trepidation didn't come so much from the adaptive cruise control, which kept the Model S following traffic ahead at a set distance, but from the self-steering, this part of Autopilot managing to keep the Model S well-centered in its lane with no help from me. Over many miles, I built up more trust in the system, letting the car do the steering in situations from bumper-to-bumper traffic and a winding road through the hills.

2016 Tesla Model S 90DEnlarge Image
Although the middle of the Model S range, the 90D offers the best range and a wealth of useful tech, such as Autopilot self-driving.
Wayne Cunningham/Roadshow
Tesla added Autopilot to its Model S line as an option last year, along with all-wheel-drive. More recently, the high-tech automaker improved its batteries, upgrading its cars from their former 65 and 85 kilowatt-hour capacity to 70 and 90 kilowatt-hour. The example I drove, the 90D, represents all these advances.

More importantly, the 90D is the current range-leader among the Model S line, boasting 288 miles on a full battery charge.

The Model S' improvements fall outside of typical automotive industry product cycles, fulfilling Tesla's promise of acting more like a technology company, constantly building and deploying new features. Tesla accomplishes that goal partially through over-the-air software updates, improving existing cars, but the 90D presents significant hardware updates over the original Model S launched four years ago.

Sit and go
Of course, this Model S exhibited the ease of use of the original. Walking up to the car with the key fob in my pocket, it automatically unlocked. When I got in the car, it powered up without me having to push a start button, so I only needed to put it in drive to get on the road.

Likewise, the design hasn't changed, its sleek, hatchback four-door body offering excellent cargo room, both front and back, and seating space. The cabin feels less cramped than most cars due to the lack of a transmission tunnel and a dashboard bare of buttons or dials.

2016 Tesla Model S 90DEnlarge Image
The flat floor in the Model S' cabin makes for enhanced passenger room.
Wayne Cunningham/Roadshow
The big, 17-inch touchscreen in the center of the dashboard shows navigation, stereo, phone, energy consumption and car settings. I easily went from full-screen to a split-screen view, the windows showing each appearing instantly. A built-in 4G/LTE data connection powers Google maps and Internet-based audio. The LCD instrument panel in front of me showed my speed, energy usage, remaining range, and intelligently swapped audio information for turn-by-turn directions when started navigation.

The instrument panel actually made the experience of driving under Autopilot more comfortable, reassuring me with graphics that showed when the Model S' sensors were detecting the lane lines and the traffic around me. Impressively, the sensors could differentiate, as shown on the screen's graphics, a passenger car from a big truck.

At speed on the freeway, Autopilot smoothly maintained the car's position in its lane, and when I took my hands off the wheel for too long, it flashed a warning on the instrument panel. In stop-and-go traffic approaching a toll booth, the car did an even better job of self-driving, recognizing traffic around it and maintaining appropriate distances.

Handling surprise
Taking over the driving myself, the ride quality proved as comfortable as any sport-luxury car, as this Model S had its optional air suspension. The electric power steering is well-tuned, turning the wheels with a quiet, natural feel and good heft.

Audi S7 vs Tesla Model S
Shootout: Audi S7 vs. Tesla Model S
Wayne Cunningham/Roadshow
The biggest surprise came when I spent the day doing laps at the Thunderhill Raceway, negotiating a series of tight, technical turns in competition with an Audi S7. I expected the Model S to get out-of-shape in the turns, but instead it proved steady and solid. The Model S' 4,647-pound curb weight made it less than ideal for a track test, but much of that weight is in the battery pack, mounted low in the chassis. That low center of gravity helped limit body roll, ensuring good grip from all four tires. In the turns, the Model S felt nicely balanced, although not entirely nimble.

Helping its grip was its native all-wheel drive, gained from having motors driving each set of wheels. The combined output of the motors comes to 417 horsepower and 485 pound-feet of torque, those numbers expressed in 0-to-60 mph times of well under 5 seconds. That thrust made for fast runs down the race track's straightaways, or simply giving me the ability to take advantage of gaps in traffic on public roads.

288 miles is more than enough for most people's daily driving needs, and if you plug in every night, you will wake up to a fully charged car every morning. The Model S makes for a far different experience than driving an internal combustion car, where you need to go to a gas station to refuel. However, longer trips in the Model S require some planning, such as scheduling stops at Tesla's free Supercharger stations.


Charging times are much lengthier than refilling a tank with gasoline. From a Level 2, 240-volt station, you get 29 miles added every hour. Tesla's Supercharger, a Level 3 charger, takes 75 minutes to fully recharge the Model S 90D's battery.

2016 Tesla Model S 90DEnlarge Image
Despite its high initial price, the Model S 90D costs less to run on a daily basis than a combustion engine car.
Wayne Cunningham/Roadshow

Low maintenance
The 2016 Tesla Model S 90D adds features to keep it competitive against the internal combustion cars in its sport luxury set. More importantly, it remains very easy to live with. In fact, the electric drivetrain should mean greatly decreased maintenance, as there are fewer moving parts. The EPA estimates that annual electricity costs for the Model S 90D should run $650, much less than buying gasoline for an equivalent internal combustion car.

Lengthy charging times mean longer trips are either out of the question or require more planning than with an internal combustion car. And while the infotainment system responds quickly to touch inputs and offers useful screens, it hasn't changed much in four years. Most notably, Tesla hasn't added any music apps beyond the ones it launched with. Along with new, useful apps, it would be nice to have some themes or other aesthetic changes to the infotainment interface.

The Model S 90D's base price of $88,000 puts it out of reach of the average buyer, and the model I drove was optioned up to around $95,000. Against its Audi, BMW and Mercedes-Benz competition, however, it makes a compelling argument, especially for its uncomplicated nature.
"""
            ]


def toothpaste_reviews():
    return [u"""
Toothpaste can do more harm than good

The next time a patient innocently asks me, “What’s the best toothpaste to use?” I’m going to unleash a whole Chunky Soup can of “You Want The Truth? You CAN’T HANDLE THE TRUTH!!!” Gosh, that’s such an overused movie quote. Sorry about that, but still.

If you’re a dental professional, isn’t this the most annoying question you get, day after day? Do you even care which toothpaste your patients use?

No. You don’t. Asking a dentist what toothpaste to use is like asking your physician which bar of soap or body scrub you should use to clean your skin. Your dentist and dental hygienist have never seen a tube of toothpaste that singlehandedly improves the health of all patients in their practice, and the reason is simple:

Toothpaste is a cosmetic.

We brush our teeth so that out mouths no longer taste like… mouth. Mouth tastes gross, right? It tastes like putrefied skin. It tastes like tongue cheese. It tastes like Cream of Barf.

On the other hand, toothpaste has been exquisitely designed to bring you a brisk rush of York Peppermint Patty, or Triple Cinnamon Heaven, or whatever flavor that drives those tubes off of the shelves in the confusing dental aisle of your local supermarket or drugstore.


Toothpaste definitely tastes better than Cream of Barf. And that’s why you use it. Not because it’s good for you. You use toothpaste because it tastes good, and because it makes you accept your mouth as part of your face again.

From a marketing perspective, all of the other things that are in your toothpaste are in there to give it additional perceived value. So let’s deconstruct these ingredients, shall we?


1. Fluoride.

This was probably the first additive to toothpaste that brought it under the jurisdiction of the Food & Drug Administration and made toothpaste part drug, part cosmetic. Over time, a fluoride toothpaste can improve the strength of teeth, but the fluoride itself does nothing to make teeth cleaner. Some people are scared of fluoride so they don’t use it. Their choice. Professionally speaking, I know that the benefits of a fluoride additive far outweigh the risks.

2. Foam.

Sodium Lauryl Sulfate is soap. Soap has a creamy, thick texture that American tongues especially like and equate to the feeling of cleanliness. There’s not enough surfactant, though, in toothpaste foam to break up the goo that grows on your teeth. If these bubbles scrubbed, you’d better believe that they would also scrub your delicate gum tissues into a bloody pulp.

3. Abrasive particles.

Most toothpastes use hydrated silica as the grit that polishes teeth. You’re probably most familiar with it as the clear beady stuff in the “Do Not Eat” packets. Depending on the size and shape of the particles, silica is the whitening ingredient in most whitening toothpastes. But whitening toothpaste cannot get your teeth any whiter than a professional dental cleaning, because it only cleans the surface. Two weeks to a whiter smile? How about 30 minutes with your hygienist? It’s much more efficient and less harsh.

4. Desensitizers.

Teeth that are sensitive to hot, cold, sweets, or a combination can benefit from the addition of potassium nitrate or stannous fluoride to a toothpaste. This is more of a palliative treatment, when the pain is the problem. Good old Time will usually make teeth feel better, too, unless the pain is coming from a cavity. Yeah, I’m talking to you, the person who is trying to heal the hole in their tooth with Sensodyne.

5. Tartar control.

It burns! It burns! If your toothpaste has a particular biting flavor, it might contain tetrasodium pyrophosphate, an ingredient that is supposed to keep calcium phosphate salts (tartar, or calculus) from fossilizing on the back of your lower front teeth. A little tartar on your teeth doesn’t harm you unless it gets really thick and you can no longer keep it clean. One problem with tartar control toothpastes is that in order for the active ingredient to work, it has to be dissolved in a stronger detergent than usual, which can affect people that are sensitive to a high pH.

6. Triclosan.

This antimicrobial is supposed to reduce infections between the gum and tooth. However, if you just keep the germs off of your teeth in the first place it’s pretty much a waste of an extra ingredient. Its safety has been questioned but, like fluoride, the bulk of the scientific research easily demonstrates that the addition of triclosan in toothpaste does much more good than harm.

Why toothpaste can be bad for you.

Let’s just say it’s not the toothpaste’s fault. It’s yours. The toothpaste is just the co-dependent enabler. You’re the one with the problem.

Remember, toothpaste is a cosmetic, first and foremost. It doesn’t clean your teeth by itself. Just in case you think I’m making this up I’ve included clinical studies in the references at the end of this article that show how ineffective toothpaste really is.

peasized

• You’re using too much.

Don’t be so suggestible! Toothpaste ads show you how to use up the tube more quickly. Just use 1/3 as much, the size of a pea. It will still taste good, I promise! And too much foam can make you lose track of where your teeth actually are located.

• You’re not taking enough time.

At least two minutes. Any less and you’re missing spots. Just ’cause it tastes better doesn’t mean you did a good job.

• You’re not paying attention.

I’ve seen people brush the same four spots for two minutes and miss the other 60% of their mouth.brushguide The toothbrush needs to touch every crevice of every tooth, not just where it lands when you go into autopilot and start thinking about what you’re going to wear that day. It’s the toothbrush friction that cleans your teeth, not the cleaning product. Plaque is a growth, like the pink or grey mildew that grows around the edges of your shower. You’ve gotta rub it off to get it off. No tooth cleaning liquid, paste, creme, gel, or powder is going to make as much of a difference as your attention to detail will.

The solution.

Use what you like. It’s that simple. If it tastes good and feels clean to you, you’ll use it more often, brush longer, feel better, be healthier.

You can use baking soda, or coconut oil, or your favorite toothpaste, or even just plain water. The key is to have a good technique and to brush often. A music video makes this demonstration a little more fun than your usual lecture at the dental office, although, in my opinion you really still need to feel what it is like to MASH THE BRISTLES OF A SOFT TOOTHBRUSH INTO YOUR GUMS:





A little more serious video from my pal Dr. Mark Burhenne where he demonstrates how to be careful with your toothbrush bristles:


Final word.

♬ It’s all about that Bass, ’bout that Bass, no bubbles. ♬ Heh, dentistry in-joke there.

Seriously, though, the bottom line is that your paste will mask brushing technique issues, so don’t put so much faith in the power of toothpaste.

Also you may have heard that some toothpastes contain decorative plastic that can get swallowed. Yeah, that was a DentalBuzz report I wrote that went viral earlier this year. And while I can’t claim total victory on that front, at least the company in question has promised that the plastic will no longer be added to their toothpaste lines very soon due to the overwhelming amount of letters, emails, and phone calls that they received as a result of people reading that article and making a difference.

But now I’m tired of talking about toothpaste.

Next topic?

I’m bringing pyorrhea back.
    """,

            u"""
I’ve been a user of Colgate Total Whitening Toothpaste for many years because I’ve always tried to maintain a healthy smile (I’m a receptionist so I need a white smile). But because I drink coffee at least twice a day (sometimes more!) and a lot of herbal teas, I’ve found that using just this toothpaste alone doesn’t really get my teeth white...

The best way to get white teeth is to really try some professional products specifically for tooth whitening. I’ve tried a few products, like Crest White Strips and found that the strips are really not as good as the trays. Although the Crest White Strips are easy to use, they really DO NOT cover your teeth perfectly like some other professional dental whitening kits. This Product did cover my teeth well however because of their custom heat trays, and whitening my teeth A LOT. I would say if you really want white teeth, use the Colgate Toothpaste and least 2 times a day, along side a professional Gel product like Shine Whitening.
    """,
            u"""
The first feature is the price, and it is right.

Next, I consider whether it will be neat to use. It is. Sometimes when I buy those new hard plastic containers, they actually get messy. Also I cannot get all the toothpaste out. It is easy to get the paste out of Colgate Total Whitening Paste without spraying it all over the cabinet.

If it does not taste good, I won't use it. Some toothpaste burns my mouth so bad that brushing my teeth is a painful experience. This one doesn't burn. It tastes simply the way toothpaste is supposed to taste.

Whitening is important. This one is supposed ot whiten. After spending money to whiten my teeth, I need a product to help ward off the bad effects of coffee and tea.

Avoiding all kinds of oral pathology is a major consideration. This toothpaste claims that it can help fight cavities, gingivitis, plaque, tartar, and bad breath.

I hope this product stays on the market a long time and does not change.
    """
            ]


if __name__ == '__main__':
    make_ecommerce_entityset()
