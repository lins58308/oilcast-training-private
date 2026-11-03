import copy
import datetime as dt
import io
import pathlib
import sys
import unittest
import zipfile

import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from collect_stability_context import aggregate_gdelt,cpc_neighbors
from stability_features import carry_features,auxiliary_features
from weekly_stability import Candidate,one_origin,train_choices


class NewContextTest(unittest.TestCase):
    def test_gdelt_uses_first_seen_not_event_date_and_masks_missing_numeric_values(self):
        lines=[]
        for i in range(100):
            row=['']*58;row[0]=str(i);row[1]='19990101';row[7]='IRN';row[17]='USA'
            row[29]='4';row[30]='';row[34]='';row[56]='20260904230000';row[57]='https://example.com/oil'
            lines.append('\t'.join(row))
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('events.CSV','\n'.join(lines))
        actual=aggregate_gdelt(data.getvalue(),dt.date(2026,9,4))
        self.assertEqual(actual['counts']['goldsteinKnown'],0)
        self.assertIsNone(actual['energyMeanTone'])
        self.assertEqual(actual['availableAt'],'2026-09-06T11:00:00Z')

    def test_neighbor_two_grade_table_keeps_other_grades_missing(self):
        html='''刊登日期：115-10-31<table><tr><th>國家</th><th>92汽油</th><th>柴油</th></tr>
        <tr><td>日本</td><td>35</td><td>32.9</td></tr><tr><td>韓國</td><td>45.2</td><td>44.9</td></tr>
        <tr><td>臺灣中油</td><td>31.2</td><td>29.9</td></tr></table>'''
        source={'id':'1','publishedDate':'2026-10-31','assumedAvailableAt':'2026-10-31T15:59:59Z',
                'gasoline92Reference':31.2,'url':'https://www.cpc.com.tw/News_Content.aspx?n=28&s=1'}
        data=cpc_neighbors(html.encode(),source)
        self.assertEqual(data['countries']['日本'],[35.,None,None,32.9])
        self.assertFalse(data['historicalHtmlVintageVerified'])
        with self.assertRaisesRegex(ValueError,'disagrees'):
            cpc_neighbors(html.replace('31.2','32.0').encode(),source)

    def test_quotes_after_the_publication_embargo_cannot_change_features(self):
        origin={'origin':'2026-09-07'}
        values={dt.date(2026,8,31)+dt.timedelta(days=i):(70+i,75+i,80+i) for i in range(5)}
        expected=carry_features([origin],values)
        values[dt.date(2026,9,5)]=(500.,500.,500.)
        np.testing.assert_array_equal(expected,carry_features([origin],values))

    def test_future_neighbor_and_risk_records_are_not_backdated_by_file_name(self):
        row={'origin':'2026-09-07','cutoff':'2026-09-06T16:00:00Z'}
        source={'neighbors':[{'availableAt':'2026-09-12T15:59:59Z'}],
                'gdelt':[{'date':'2026-09-04','availableAt':'2026-09-06T23:00:00Z'}]}
        got=auxiliary_features([row],source,np.ones((1,4))*30.)
        self.assertFalse(got.any())


class WeeklyRefitTest(unittest.TestCase):
    def test_future_labels_do_not_change_forecast(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(140,6));y=rng.normal(size=(140,2))*.2
        data={h:{'base':x,'carry':x,'enriched':x,'policyCovered':np.zeros(140,dtype=bool)} for h in range(2)}
        spec=Candidate('test','robust',False)
        expected=one_origin(data,y,120,spec);y[120:]=100
        np.testing.assert_array_equal(expected,one_origin(data,y,120,spec))

    def test_average_win_cannot_hide_diesel_regression(self):
        base={f:{'mae':.3,'directionAccuracy':.5,'falseMoveRate':0.} for f in ('92','95','98','diesel')}
        good={f:{'mae':.2,'directionAccuracy':.6,'falseMoveRate':.1} for f in base}
        history={'candidate':[copy.deepcopy(good) for _ in range(7)]};baseline=[copy.deepcopy(base) for _ in range(7)]
        history['candidate'][-1]['diesel']['mae']=.31
        choice=train_choices(history,baseline,'v7')
        self.assertEqual(choice[0]['selected'],'candidate')
        self.assertEqual(choice[1]['selected'],'unchanged')

    def test_overlapping_summer_window_is_not_independent_confirmation(self):
        base={f:{'mae':.3,'directionAccuracy':.5,'falseMoveRate':0.} for f in ('92','95','98','diesel')}
        good={f:{'mae':.2,'directionAccuracy':.6,'falseMoveRate':.1} for f in base}
        history={'candidate':[copy.deepcopy(good) for _ in range(7)]};baseline=[copy.deepcopy(base) for _ in range(7)]
        for k in (2,3):
            for f in base:history['candidate'][k][f]['mae']=.3
        choice=train_choices(history,baseline,'v6')
        self.assertEqual(choice[0]['selected'],'unchanged')

    def test_three_wins_do_not_hide_a_regression_in_the_fourth_independent_era(self):
        base={f:{'mae':.3,'directionAccuracy':.5,'falseMoveRate':0.} for f in ('92','95','98','diesel')}
        good={f:{'mae':.2,'directionAccuracy':.6,'falseMoveRate':.1} for f in base}
        history={'candidate':[copy.deepcopy(good) for _ in range(7)]};baseline=[copy.deepcopy(base) for _ in range(7)]
        for f in base:history['candidate'][2][f]['mae']=.31
        self.assertEqual(train_choices(history,baseline,'v7')[0]['selected'],'unchanged')


if __name__=='__main__':unittest.main()
