import unittest
from unittest.mock import patch
from research.collect_cpc_policy import announcement, FUEL_PAIR, TWO_AMOUNTS

HTML="""<html><body><article><ul><li><span>刊登日期：115-02-15</span></li></ul>
<p>台灣中油公司自明（16）日凌晨零時起汽、柴油價格皆不予調整，參考零售價格分別為92無鉛汽油每公升27.2元、95無鉛汽油每公升28.7元、98無鉛汽油每公升30.7元、超級柴油每公升26.6元。</p>
<p>按浮動油價機制調整原則，汽、柴油價格應各調漲1.2元及0.4元；因應春節期間只跌不漲政策，汽柴油均不予調整。</p>
<table><tr><th>吸收金額</th><th>政府調降貨物稅</th><th>維持亞鄰最低價</th><th>平穩措施</th><th>合計</th></tr>
<tr><td>汽油</td><td>2.2</td><td>1.2</td><td>0.0</td><td>3.4</td></tr>
<tr><td>柴油</td><td>1.6</td><td>0.4</td><td>0.0</td><td>2.0</td></tr></table>
<p>相關檔案</p></article></body></html>"""


class CpcAnnouncementTest(unittest.TestCase):
    def test_extract_actual_policy_components(self):
        with patch("research.collect_cpc_policy.http_text",return_value=HTML):
            a=announcement(112027,"維持亞鄰最低價 汽、柴油價格不調整")
        self.assertEqual(a["publishedDate"],"2026-02-15")
        self.assertEqual(a["assumedAvailableAt"],"2026-02-15T15:59:59Z")
        self.assertEqual(a["theoreticalGas"],1.2)
        self.assertEqual(a["theoreticalDiesel"],.4)
        self.assertEqual(a["absorptionGas"],3.4)
        self.assertEqual(a["absorptionDiesel"],2.0)
        self.assertEqual(a["taxReliefGas"],2.2)
        self.assertEqual(a["taxReliefDiesel"],1.6)
        self.assertTrue(a["bothFuelFrozen"])
        self.assertTrue(a["holidayNoRise"])

    def test_explicit_pair_regex_is_narrow(self):
        self.assertIsNotNone(FUEL_PAIR.search("汽、柴油價格應各調漲1.2元及0.4元"))
        self.assertIsNotNone(TWO_AMOUNTS.search("汽、柴油各共吸收3.4元及2.0元"))
        self.assertIsNone(FUEL_PAIR.search("天然氣價格應調漲9元"))

if __name__=="__main__":unittest.main()
