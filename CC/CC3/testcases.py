import random
import unittest
from solution import calculate
from character_data import *
from testdata import *
from xml.dom import minidom


class CC3TestCases(unittest.TestCase):

    def test_monotone(self):
        # Monotone increasing power levels

        # Case 0
        # Power levels [1047, 1083, 1096, 1193, 1207, 1265, 1435, 1670]
        characters0 = ['Onizuka, Eikichi', 'Natsuki, Subaru', 'Yukihira, Souma', 'Akemi, Homura',
                       'Usui, Takumi',
                       'Kurapika', 'Yuuki, Asuna', 'Son, Gokuu']
        expected0 = [('Onizuka, Eikichi', 0), ('Natsuki, Subaru', 0), ('Yukihira, Souma', 0),
                     ('Akemi, Homura', 0),
                     ('Usui, Takumi', 0), ('Kurapika', 0), ('Yuuki, Asuna', 0), ('Son, Gokuu', 0)]
        result0 = calculate(characters0, character_details_0)
        self.assertEqual(expected0, result0)

        # Monotone decreasing power levels

        # Case 1
        # Power levels [1670, 1435, 1265, 1207, 1193, 1096, 1083, 1047]
        characters1 = ['Son, Gokuu', 'Yuuki, Asuna', 'Kurapika', 'Usui, Takumi', 'Akemi, Homura',
                       'Yukihira, Souma',
                       'Natsuki, Subaru', 'Onizuka, Eikichi']
        expected1 = [('Son, Gokuu', 8326), ('Yuuki, Asuna', 6891), ('Kurapika', 5626),
                     ('Usui, Takumi', 4419),
                     ('Akemi, Homura', 3226), ('Yukihira, Souma', 2130), ('Natsuki, Subaru', 1047),
                     ('Onizuka, Eikichi', 0)]
        result1 = calculate(characters1, character_details_1)
        self.assertEqual(expected1, result1)

    def test_pyramid(self):
        # Monotone increasing followed by monotone decreasing

        # Case 2
        #Power levels [1047, 1083, 1096, 1193, 1207, 1265, 1435, 1670, 1435, 1265, 1207, 1193, 1096, 1083, 1047]
        characters2 = ['Onizuka, Eikichi', 'Natsuki, Subaru', 'Yukihira, Souma', 'Akemi, Homura',
                       'Usui, Takumi',
                       'Kurapika', 'Yuuki, Asuna', 'Son, Gokuu', 'Yuuki, Asuna', 'Kurapika',
                       'Usui, Takumi',
                       'Akemi, Homura', 'Yukihira, Souma', 'Natsuki, Subaru', 'Onizuka, Eikichi']
        expected2 = [('Onizuka, Eikichi', 0), ('Natsuki, Subaru', 0), ('Yukihira, Souma', 0),
                     ('Akemi, Homura', 0),
                     ('Usui, Takumi', 0), ('Kurapika', 0), ('Yuuki, Asuna', 0),
                     ('Son, Gokuu', 8326),
                     ('Yuuki, Asuna', 6891), ('Kurapika', 5626), ('Usui, Takumi', 4419),
                     ('Akemi, Homura', 3226),
                     ('Yukihira, Souma', 2130), ('Natsuki, Subaru', 1047), ('Onizuka, Eikichi', 0)]
        result2 = calculate(characters2, character_details_2)
        self.assertEqual(expected2, result2)

        # Monotone decreasing followed by monotone increasing

        # Case 3
        #Power levels [1670, 1435, 1265, 1207, 1193, 1096, 1083, 1047, 1083, 1096, 1193, 1207, 1265, 1435, 1670]
        characters3 = ['Son, Gokuu', 'Yuuki, Asuna', 'Kurapika', 'Usui, Takumi', 'Akemi, Homura',
                       'Yukihira, Souma',
                       'Natsuki, Subaru', 'Onizuka, Eikichi', 'Natsuki, Subaru', 'Yukihira, Souma',
                       'Akemi, Homura',
                       'Usui, Takumi', 'Kurapika', 'Yuuki, Asuna', 'Son, Gokuu']
        expected3 = [('Son, Gokuu', 17275), ('Yuuki, Asuna', 14170), ('Kurapika', 11470),
                     ('Usui, Takumi', 8998),
                     ('Akemi, Homura', 6598), ('Yukihira, Souma', 4309), ('Natsuki, Subaru', 2130),
                     ('Onizuka, Eikichi', 0), ('Natsuki, Subaru', 0), ('Yukihira, Souma', 0),
                     ('Akemi, Homura', 0),
                     ('Usui, Takumi', 0), ('Kurapika', 0), ('Yuuki, Asuna', 0), ('Son, Gokuu', 0)]
        result3 = calculate(characters3, character_details_3)
        self.assertEqual(expected3, result3)

    def test_intermediate(self):
        # Varied, English input only, under 9000

        # Case 4
        # Power levels [1207, 6677, 8929, 3711, 4563, 2063, 6750, 4076, 7127, 4897, 8529, 8861, 6040, 1816, 2900, 1435, 8607, 7040, 3099, 8374]
        characters4 = ['Usui, Takumi', 'Toosaka, Rin', 'Sanji', 'Reigen, Arataka', 'Saber',
                       'Yagami, Light',
                       'Michaelis, Sebastian', 'C.C.', 'Miyazono, Kaori', 'Tempest, Rimuru',
                       'Zero Two', 'Emilia',
                       'Morow, Hisoka', 'Kuujou, Joutarou', 'Saiki, Kusuo', 'Yuuki, Asuna',
                       'Kamado, Tanjirou',
                       'Spiegel, Spike', 'Tokisaki, Kurumi', 'Sakata, Gintoki']
        expected4 = [('Usui, Takumi', 0), ('Toosaka, Rin', 0), ('Sanji', 89888),
                     ('Reigen, Arataka', 0), ('Saber', 2063),
                     ('Yagami, Light', 0), ('Michaelis, Sebastian', 4076), ('C.C.', 0),
                     ('Miyazono, Kaori', 4897),
                     ('Tempest, Rimuru', 0), ('Zero Two', 0), ('Emilia', 39311),
                     ('Morow, Hisoka', 6151),
                     ('Kuujou, Joutarou', 0), ('Saiki, Kusuo', 1435), ('Yuuki, Asuna', 0),
                     ('Kamado, Tanjirou', 18513),
                     ('Spiegel, Spike', 3099), ('Tokisaki, Kurumi', 0), ('Sakata, Gintoki', 0)]
        result4 = calculate(characters4, character_details_4)
        self.assertEqual(expected4, result4)

        # Case 5
        # Power levels [1193, 6120, 2929, 1265, 2454, 1670, 1047, 2075, 2582, 5853, 2063, 3297, 5697, 2900, 2929, 8236, 7040, 7040, 8936, 5615]
        characters5 = ['Akemi, Homura', 'Fujiwara, Chika', 'Megumin', 'Kurapika', 'Yeager, Eren',
                       'Son, Gokuu',
                       'Onizuka, Eikichi', 'Liebert, Johan', 'Senkuu', 'Arlert, Armin',
                       'Yagami, Light',
                       'Uchiha, Itachi', 'Uchiha, Sasuke', 'Saiki, Kusuo', 'Megumin',
                       'Heiwajima, Shizuo',
                       'Spiegel, Spike', 'Spiegel, Spike', 'Midoriya, Izuku', 'Kaneki, Ken']
        expected5 = [('Akemi, Homura', 0), ('Fujiwara, Chika', 36761), ('Megumin', 11093),
                     ('Kurapika', 0),
                     ('Yeager, Eren', 4792), ('Son, Gokuu', 1047), ('Onizuka, Eikichi', 0),
                     ('Liebert, Johan', 0),
                     ('Senkuu', 0), ('Arlert, Armin', 16886), ('Yagami, Light', 0),
                     ('Uchiha, Itachi', 0),
                     ('Uchiha, Sasuke', 5829), ('Saiki, Kusuo', 0), ('Megumin', 0),
                     ('Heiwajima, Shizuo', 14080),
                     ('Spiegel, Spike', 7040), ('Spiegel, Spike', 0), ('Midoriya, Izuku', 5615),
                     ('Kaneki, Ken', 0)]
        result5 = calculate(characters5, character_details_5)
        self.assertEqual(expected5, result5)

        # Case 6
        # Power levels [5287, 7642, 3345, 7394, 3099, 6998, 8929, 3611, 5287, 6715, 6503, 4076, 2450, 4897, 4682, 1265, 3297, 2133, 5131, 8101]
        characters6 = ['Zoë, Hange', 'Kurosaki, Ichigo', 'Holo', 'Kirigaya, Kazuto',
                       'Tokisaki, Kurumi',
                       'Makise, Kurisu', 'Sanji', 'Ackerman, Mikasa', 'Zoë, Hange', 'Koro-sensei',
                       'Lamperouge, Lelouch',
                       'C.C.', 'Askeladd', 'Tempest, Rimuru', 'Saitama', 'Kurapika',
                       'Uchiha, Itachi', 'Shiina, Mashiro',
                       'Yukinoshita, Yukino', 'Lawliet, L']
        expected6 = [('Zoë, Hange', 0), ('Kurosaki, Ichigo', 20836), ('Holo', 0),
                     ('Kirigaya, Kazuto', 10097),
                     ('Tokisaki, Kurumi', 0), ('Makise, Kurisu', 0), ('Sanji', 58148),
                     ('Ackerman, Mikasa', 0),
                     ('Zoë, Hange', 0), ('Koro-sensei', 34434), ('Lamperouge, Lelouch', 27931),
                     ('C.C.', 2450),
                     ('Askeladd', 0), ('Tempest, Rimuru', 11377), ('Saitama', 6695),
                     ('Kurapika', 0),
                     ('Uchiha, Itachi', 2133), ('Shiina, Mashiro', 0), ('Yukinoshita, Yukino', 0),
                     ('Lawliet, L', 0)]
        result6 = calculate(characters6, character_details_6)
        self.assertEqual(expected6, result6)

        # Case 7
        # Power levels [7243, 7195, 8236, 4378, 7127, 8619, 8935, 7187, 6910, 5342, 5697, 6503, 8251, 1047, 5287, 3919, 5615, 7187, 7120, 3156]
        characters7 = ['Freecss, Gon', 'Nakano, Miku', 'Heiwajima, Shizuo', 'Elric, Edward',
                       'Miyazono, Kaori',
                       'Hikigaya, Hachiman', 'Todoroki, Shouto', 'Senjougahara, Hitagi',
                       'Okabe, Rintarou',
                       'Gasai, Yuno', 'Uchiha, Sasuke', 'Lamperouge, Lelouch', 'Gremory, Rias',
                       'Onizuka, Eikichi',
                       'Zoë, Hange', 'Hinata, Shouyou', 'Kaneki, Ken', 'Senjougahara, Hitagi',
                       'Gojou, Satoru',
                       'Satou, Kazuma']
        expected7 = [('Freecss, Gon', 7195), ('Nakano, Miku', 0), ('Heiwajima, Shizuo', 11505),
                     ('Elric, Edward', 0),
                     ('Miyazono, Kaori', 0), ('Hikigaya, Hachiman', 0), ('Todoroki, Shouto', 73221),
                     ('Senjougahara, Hitagi', 24452), ('Okabe, Rintarou', 17542),
                     ('Gasai, Yuno', 0),
                     ('Uchiha, Sasuke', 0), ('Lamperouge, Lelouch', 0), ('Gremory, Rias', 33331),
                     ('Onizuka, Eikichi', 0), ('Zoë, Hange', 3919), ('Hinata, Shouyou', 0),
                     ('Kaneki, Ken', 0),
                     ('Senjougahara, Hitagi', 10276), ('Gojou, Satoru', 3156), ('Satou, Kazuma', 0)]
        result7 = calculate(characters7, character_details_7)
        self.assertEqual(expected7, result7)

    def test_over_9000(self):
        # Some characters with power level over 9000, English input only

        # Case 8
        # Power levels [4682, 9111, 9977, 1694, 7187, 8990, 9001, 8936, 2063, 9001, 9919, 9919, 9977, 9848, 8936, 9919, 2582, 7394, 9919, 6998]
        characters8 = ['Saitama', 'Sora', 'Kageyama, Shigeo', 'Aisaka, Taiga',
                       'Senjougahara, Hitagi', 'Revy', 'Vegeta',
                       'Midoriya, Izuku', 'Yagami, Light', 'Vegeta', 'Hei', 'Hei',
                       'Kageyama, Shigeo', 'Levi',
                       'Midoriya, Izuku', 'Hei', 'Senkuu', 'Kirigaya, Kazuto', 'Hei',
                       'Makise, Kurisu']
        expected8 = [('Saitama', 0), ('空', 0), ('影山 茂夫', 132283), ('Aisaka, Taiga', 0),
                     ('Senjougahara, Hitagi', 0),
                     ('Revy', 0), ('ベジータ', 20000), ('Midoriya, Izuku', 2063), ('Yagami, Light', 0),
                     ('ベジータ', 0),
                     ('黒（ヘイ）', 9919), ('黒（ヘイ）', 0), ('影山 茂夫', 55596), ('リヴァイ', 8936),
                     ('Midoriya, Izuku', 0),
                     ('黒（ヘイ）', 26893), ('Senkuu', 0), ('Kirigaya, Kazuto', 0), ('黒（ヘイ）', 6998),
                     ('Makise, Kurisu', 0)]
        result8 = calculate(characters8, character_details_8)
        self.assertEqual(expected8, result8)

        # Case 9
        # Power levels [3345, 8990, 9848, 9111, 8607, 9894, 2063, 6380, 9111, 6380, 4682, 4239, 9894, 9977, 9919, 9894, 1096, 9159, 9393, 2133]
        characters9 = ['Holo', 'Revy', 'Levi', 'Sora', 'Kamado, Tanjirou', 'Orihara, Izaya',
                       'Yagami, Light', 'Guts',
                       'Sora', 'Guts', 'Saitama', 'Mustang, Roy', 'Orihara, Izaya',
                       'Kageyama, Shigeo', 'Hei',
                       'Orihara, Izaya', 'Yukihira, Souma', 'Takanashi, Rikka', 'Emiya, Kiritsugu',
                       'Shiina, Mashiro']
        expected9 = [('Holo', 0), ('Revy', 0), ('リヴァイ', 17718), ('空', 8607),
                     ('Kamado, Tanjirou', 0), ('折原 臨也', 42749),
                     ('Yagami, Light', 0), ('Guts', 0), ('空', 15301), ('Guts', 8921),
                     ('Saitama', 4239),
                     ('Mustang, Roy', 0), ('折原 臨也', 0), ('影山 茂夫', 41594), ('黒（ヘイ）', 31675),
                     ('折原 臨也', 21781),
                     ('Yukihira, Souma', 0), ('小鳥遊 六花', 0), ('衛宮 切嗣', 2133), ('Shiina, Mashiro', 0)]
        result9 = calculate(characters9, character_details_9)
        self.assertEqual(expected9, result9)

        # Case 10
        # Power levels [2900, 5918, 9159, 9393, 9977, 9159, 9923, 9159, 6681, 8374, 2075, 8861, 6715, 5853, 3345, 9923, 9159, 9159, 9111, 6677]
        characters10 = ['Saiki, Kusuo', 'Misaka, Mikoto', 'Takanashi, Rikka', 'Emiya, Kiritsugu',
                        'Kageyama, Shigeo',
                        'Takanashi, Rikka', 'Zoldyck, Killua', 'Takanashi, Rikka', 'Kamina',
                        'Sakata, Gintoki',
                        'Liebert, Johan', 'Emilia', 'Koro-sensei', 'Arlert, Armin', 'Holo',
                        'Zoldyck, Killua',
                        'Takanashi, Rikka', 'Takanashi, Rikka', 'Sora', 'Toosaka, Rin']
        expected10 = [('Saiki, Kusuo', 0), ('Misaka, Mikoto', 0), ('小鳥遊 六花', 0), ('衛宮 切嗣', 0),
                      ('影山 茂夫', 114174),
                      ('小鳥遊 六花', 0), ('キルア=ゾルディック', 95092), ('小鳥遊 六花', 41904), ('Kamina', 0),
                      ('Sakata, Gintoki', 2075),
                      ('Liebert, Johan', 0), ('Emilia', 15913), ('Koro-sensei', 9198),
                      ('Arlert, Armin', 3345),
                      ('Holo', 0), ('キルア=ゾルディック', 34106), ('小鳥遊 六花', 24947), ('小鳥遊 六花', 15788),
                      ('空', 6677),
                      ('Toosaka, Rin', 0)]
        result10 = calculate(characters10, character_details_10)
        self.assertEqual(expected10, result10)

        # Case 11
        # Power levels [9280, 6120, 9159, 9894, 6681, 9848, 7642, 9848, 4963, 5031, 6910, 1265, 9001, 1207, 6750, 8829, 9894, 9977, 9001, 9848]
        characters11 = ['Araragi, Koyomi', 'Fujiwara, Chika', 'Takanashi, Rikka', 'Orihara, Izaya',
                        'Kamina', 'Levi',
                        'Kurosaki, Ichigo', 'Levi', 'Souryuu, Asuka Langley', 'Dragneel, Natsu',
                        'Okabe, Rintarou',
                        'Kurapika', 'Vegeta', 'Usui, Takumi', 'Michaelis, Sebastian',
                        'Akabane, Karma', 'Orihara, Izaya',
                        'Kageyama, Shigeo', 'Vegeta', 'Levi']
        expected11 = [('阿良々木 暦', 15279), ('Fujiwara, Chika', 0), ('小鳥遊 六花', 0), ('折原 臨也', 87869),
                      ('Kamina', 0),
                      ('リヴァイ', 61446), ('Kurosaki, Ichigo', 0), ('リヴァイ', 43956),
                      ('Souryuu, Asuka Langley', 0),
                      ('Dragneel, Natsu', 0), ('Okabe, Rintarou', 1265), ('Kurapika', 0),
                      ('ベジータ', 16786),
                      ('Usui, Takumi', 0), ('Michaelis, Sebastian', 0), ('Akabane, Karma', 0),
                      ('折原 臨也', 0),
                      ('影山 茂夫', 18849), ('ベジータ', 0), ('リヴァイ', 0)]
        result11 = calculate(characters11, character_details_11)
        self.assertEqual(expected11, result11)

    def test_mixed_input_language(self):
        # Input in English/Japanese, might be over 9000

        # Case 12
        # Power levels [6715, 3156, 1670, 5342, 1670, 5853, 1816, 6715, 3156, 7394, 2454, 9894, 8236, 4873, 5615, 9977, 6503, 6040, 1096, 2063]
        characters12 = ['殺せんせー', 'Satou, Kazuma', 'Son, Gokuu', 'Gasai, Yuno', 'Son, Gokuu',
                        'アルミン・アルレルト', '空条 承太郎',
                        'Koro-sensei', 'Satou, Kazuma', '桐ヶ谷 和人', 'エレン・イェーガー', '折原 臨也',
                        'Heiwajima, Shizuo', 'Alucard',
                        'Kaneki, Ken', 'Kageyama, Shigeo', 'Lamperouge, Lelouch', 'Morow, Hisoka',
                        '幸平 創真', '夜神 月']
        expected12 = [('Koro-sensei', 29378), ('Satou, Kazuma', 1670), ('Son, Gokuu', 0),
                      ('Gasai, Yuno', 1670),
                      ('Son, Gokuu', 0), ('Arlert, Armin', 1816), ('Kuujou, Joutarou', 0),
                      ('Koro-sensei', 3156),
                      ('Satou, Kazuma', 0), ('Kirigaya, Kazuto', 2454), ('Yeager, Eren', 0),
                      ('折原 臨也', 18724),
                      ('Heiwajima, Shizuo', 10488), ('Alucard', 0), ('Kaneki, Ken', 0),
                      ('影山 茂夫', 15702),
                      ('Lamperouge, Lelouch', 9199), ('Morow, Hisoka', 3159),
                      ('Yukihira, Souma', 0),
                      ('Yagami, Light', 0)]
        result12 = calculate(characters12, character_details_12)
        self.assertEqual(expected12, result12)

        # Case 13
        # Power levels [8990, 4239, 2075, 2063, 4682, 3099, 8861, 6998, 9393, 5853, 7187, 3611, 1193, 7127, 7169, 4873, 1096, 8236, 6120, 7195]
        characters13 = ['Revy', 'Mustang, Roy', 'ヨハン・リーベルト', '夜神 月', 'サイタマ', '時崎 狂三', 'エミリア',
                        'Makise, Kurisu', '衛宮 切嗣',
                        'Arlert, Armin', 'Senjougahara, Hitagi', 'Ackerman, Mikasa', '暁美 ほむら',
                        'Miyazono, Kaori', 'Rem',
                        'Alucard', '幸平 創真', '平和島 静雄', '藤原 千花', '中野 三玖']
        expected13 = [('Revy', 32017), ('Mustang, Roy', 4138), ('Liebert, Johan', 2063),
                      ('Yagami, Light', 0),
                      ('Saitama', 3099), ('Tokisaki, Kurumi', 0), ('Emilia', 6998),
                      ('Makise, Kurisu', 0),
                      ('衛宮 切嗣', 59660), ('Arlert, Armin', 0), ('Senjougahara, Hitagi', 25069),
                      ('Ackerman, Mikasa', 1193),
                      ('Akemi, Homura', 0), ('Miyazono, Kaori', 0), ('Rem', 5969),
                      ('Alucard', 1096),
                      ('Yukihira, Souma', 0), ('Heiwajima, Shizuo', 13315), ('Fujiwara, Chika', 0),
                      ('Nakano, Miku', 0)]
        result13 = calculate(characters13, character_details_13)
        self.assertEqual(expected13, result13)

        # Case 14
        # Power levels [8018, 1207, 4076, 8101, 3244, 4076, 3297, 5432, 8529, 5131, 8990, 2929, 8236, 3244, 4076, 7195, 4963, 8374, 2450, 4873]
        characters14 = ['ロロノア・ゾロ', 'Usui, Takumi', 'シー・ツー', 'Lawliet, L', 'エルヴィン・スミス', 'C.C.',
                        'Uchiha, Itachi', '爆豪 勝己',
                        'ゼロツー', '雪ノ下 雪乃', 'レヴィ', 'Megumin', 'Heiwajima, Shizuo', 'エルヴィン・スミス',
                        'C.C.', 'Nakano, Miku',
                        '惣流・アスカ・ラングレー', 'Sakata, Gintoki', 'Askeladd', 'アーカード']
        expected14 = [('Roronoa, Zoro', 5283), ('Usui, Takumi', 0), ('C.C.', 0),
                      ('Lawliet, L', 16049),
                      ('Smith, Erwin', 0), ('C.C.', 3297), ('Uchiha, Itachi', 0),
                      ('Bakugou, Katsuki', 0),
                      ('Zero Two', 5131), ('Yukinoshita, Yukino', 0), ('Revy', 46340),
                      ('Megumin', 0),
                      ('Heiwajima, Shizuo', 19478), ('Smith, Erwin', 0), ('C.C.', 0),
                      ('Nakano, Miku', 4963),
                      ('Souryuu, Asuka Langley', 0), ('Sakata, Gintoki', 7323), ('Askeladd', 0),
                      ('Alucard', 0)]
        result14 = calculate(characters14, character_details_14)
        self.assertEqual(expected14, result14)

        # Case 15
        # Power levels [6380, 6910, 8529, 3155, 8929, 8251, 9000, 7169, 9111, 3711, 2157, 2063, 2450, 1670, 3828, 4378, 6998, 8990, 2454, 3297]
        characters15 = ['Guts', 'Okabe, Rintarou', 'ゼロツー', 'Trafalgar, Law', 'Sanji', 'リアス・グレモリー',
                        'Joestar, Joseph😉',
                        'レム', '空', '霊幻 新隆', 'Ayanokouji, Kiyotaka', '夜神 月', 'アシェラッド', 'Son, Gokuu',
                        'Monkey D., Luffy',
                        'Elric, Edward', 'Makise, Kurisu', 'レヴィ', 'Yeager, Eren', 'Uchiha, Itachi']
        expected15 = [('Guts', 0), ('Okabe, Rintarou', 0), ('Zero Two', 3155),
                      ('Trafalgar, Law', 0), ('Sanji', 8251),
                      ('Gremory, Rias', 0), ('Joestar, Joseph😉', 7169), ('Rem', 0), ('空', 41996),
                      ('Reigen, Arataka', 8340), ('Ayanokouji, Kiyotaka', 2063),
                      ('Yagami, Light', 0), ('Askeladd', 1670),
                      ('Son, Gokuu', 0), ('Monkey D., Luffy', 0), ('Elric, Edward', 0),
                      ('Makise, Kurisu', 0),
                      ('Revy', 5751), ('Yeager, Eren', 0), ('Uchiha, Itachi', 0)]
        result15 = calculate(characters15, character_details_15)
        self.assertEqual(expected15, result15)

    def test_comprehensive(self):
        # Large, random test cases

        random.seed(331)

        # Case 16
        characters16 = random.choices(character_names, k=1000)

        result16 = calculate(characters16, character_details_16)
        self.assertEqual(expected16, result16)

        # Case 17
        characters17 = random.choices(character_names, k=1000)

        result17 = calculate(characters17, character_details_17)
        self.assertEqual(expected17, result17)

        # Case 18
        characters18 = random.choices(character_names, k=1000)

        result18 = calculate(characters18, character_details_18)
        self.assertEqual(expected18, result18)

        # Case 19
        characters19 = random.choices(character_names, k=1000)

        result19 = calculate(characters19, character_details_19)
        self.assertEqual(expected19, result19)

    def test_README(self):
        path = "CC3/README.xml"
        xml_doc = minidom.parse(path)
        response = {}
        tags = ["netid", "feedback", "difficulty", "time", "citations", "type", "number"]

        # Assert that we can access all tags
        for tag in tags:
            raw = xml_doc.getElementsByTagName(tag)[0].firstChild.nodeValue
            lines = [s.strip() for s in raw.split("\n")]  # If multiple lines, strip each line
            clean = " ".join(lines).strip()  # Rejoin lines with spaces and strip leading space
            self.assertNotEqual("REPLACE", clean)  # Make sure entry was edited
            response[tag] = clean  # Save each entry

        # Assert that difficulty is a float between 0-10
        difficulty_float = float(response["difficulty"])
        self.assertGreaterEqual(difficulty_float, 0.0)
        self.assertLessEqual(difficulty_float, 10.0)

        # Assert that hours is a float between 0-100 (hopefully it didn't take 100 hours!)
        time_float = float(response["time"])
        self.assertGreaterEqual(time_float, 0.0)
        self.assertLessEqual(time_float, 100.0)

        # Assert assignment type and number was not changed
        self.assertEqual("CC", response["type"])
