import base64,hashlib,json,pathlib,tempfile,unittest,sys
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import pipeline

class FakeGitHub:
    repository='lins58308/oilcast-models'
    def __init__(self):self.calls=[];self.releases={};self.next_id=1;self.fail_asset=None;self.latest_value=None
    def release(self,tag):return self.releases.get(tag)
    def asset(self,release,name):
        for asset in release.get('assets',[]):
            if asset['name']==name:return asset['data']
        raise RuntimeError('missing asset')
    def upload(self,release,name,data,replace=False):
        self.calls.append(('upload',release['tag_name'],name))
        if name==self.fail_asset:raise RuntimeError('simulated upload failure')
        release.setdefault('assets',[])[:]=[a for a in release.get('assets',[]) if a['name']!=name]
        release['assets'].append({'name':name,'data':data});return {'size':len(data)}
    def request(self,path,method='GET',value=None,**kw):
        self.calls.append((method,path,dict(value or {})))
        if method=='POST' and path=='/releases':
            row=dict(value,id=self.next_id,assets=[]);self.next_id+=1;self.releases[value['tag_name']]=row;return row
        if method=='PATCH':
            row=next(r for r in self.releases.values() if r['id']==int(path.rsplit('/',1)[1]));row.update(value)
            if not row['draft'] and value.get('make_latest')=='true':self.latest_value=row
            return row
        raise AssertionError('Unexpected fake operation')

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.folder=pathlib.Path(self.temp.name);self.api=FakeGitHub();self.model={'rows':[],'multisource':{'news':[]},'model':{'trainedAt':'2026-10-12T03:00:00Z','revision':2}}
        data=json.dumps(self.model).encode();self.manifest={'demo':False,'trainedAt':'2026-10-12T03:00:00Z','payloadSha256':hashlib.sha256(data).hexdigest(),'payloadBytes':len(data),'payloadUrl':'https://github.com/lins58308/oilcast-models/releases/download/model-fixture/model.json'}
        # Transport/order fixture only. The production main() separately performs
        # Java RSA and full model validation before calling publish(). No network here.
        (self.folder/'model.json').write_bytes(data);self.write_envelope();(self.folder/'checkpoint.json').write_text(json.dumps({'cloudStateSchema':1,**self.model}))
    def tearDown(self):self.temp.cleanup()
    def write_envelope(self):(self.folder/'envelope.json').write_text(json.dumps({'manifestBase64':base64.b64encode(json.dumps(self.manifest).encode()).decode()}))
    def test_publish_assets_before_latest(self):
        row=pipeline.publish(self.api,'model-fixture',self.folder)
        self.assertFalse(row['draft']);self.assertIs(self.api.latest_value,row)
        self.assertEqual([x[2] for x in self.api.calls if x[0]=='upload'],['model.json','envelope.json','checkpoint.json'])
        patch=next(i for i,x in enumerate(self.api.calls) if x[0]=='PATCH')
        self.assertTrue(all(i<patch for i,x in enumerate(self.api.calls) if x[0]=='upload' and x[2] in {'model.json','envelope.json'}))
    def test_second_asset_failure_keeps_model_unpublished(self):
        self.api.fail_asset='envelope.json'
        with self.assertRaises(RuntimeError):pipeline.publish(self.api,'model-fixture',self.folder)
        self.assertIsNone(self.api.latest_value);self.assertTrue(self.api.releases['model-fixture']['draft']);self.assertFalse(any(x[0]=='PATCH' for x in self.api.calls))
    def test_demo_fails_without_mutations(self):
        self.manifest['demo']=True;self.write_envelope()
        with self.assertRaises(RuntimeError):pipeline.publish(self.api,'model-fixture',self.folder)
        self.assertEqual(self.api.calls,[])
    def test_corruption_fails_without_mutations(self):
        (self.folder/'model.json').write_bytes(b'changed')
        with self.assertRaises(RuntimeError):pipeline.publish(self.api,'model-fixture',self.folder)
        self.assertEqual(self.api.calls,[])
    def test_wrong_repository_fails(self):
        self.manifest['payloadUrl']='https://github.com/other/repo/releases/download/model-fixture/model.json';self.write_envelope()
        with self.assertRaises(RuntimeError):pipeline.publish(self.api,'model-fixture',self.folder)
        self.assertEqual(self.api.calls,[])
    def test_immutable_release_is_not_overwritten(self):
        self.api.releases['model-fixture']={'tag_name':'model-fixture'}
        with self.assertRaises(RuntimeError):pipeline.publish(self.api,'model-fixture',self.folder)
        self.assertEqual(self.api.calls,[])
    def test_checkpoint_failure_preserves_published_model_and_recovers(self):
        self.api.fail_asset='checkpoint.json'
        with self.assertRaises(RuntimeError):pipeline.publish(self.api,'model-fixture',self.folder)
        self.assertIsNotNone(self.api.latest_value)
        self.api.releases.pop('training-checkpoint');out=self.folder/'restored.json'
        pipeline.restore_checkpoint(self.api,self.api.latest_value,out)
        self.assertEqual(json.loads(out.read_text())['model']['revision'],2)
    def test_newer_publication_repairs_older_checkpoint(self):
        row=pipeline.publish(self.api,'model-fixture',self.folder);old={'cloudStateSchema':1,**self.model,'model':{'trainedAt':'2026-10-11T03:00:00Z','revision':1}}
        self.api.releases['training-checkpoint']['assets'][0]['data']=json.dumps(old).encode();out=self.folder/'restored.json';pipeline.restore_checkpoint(self.api,row,out)
        self.assertEqual(json.loads(out.read_text())['model']['revision'],2)
    def test_no_checkpoint_first_run_uses_seed(self):
        out=self.folder/'absent.json';pipeline.restore_checkpoint(self.api,None,out);self.assertFalse(out.exists())
    def test_auth_failure_does_not_reset_checkpoint(self):
        self.api.release=lambda tag:(_ for _ in ()).throw(pipeline.ApiError(403));out=self.folder/'absent.json'
        with self.assertRaises(pipeline.ApiError):pipeline.restore_checkpoint(self.api,None,out)
        self.assertFalse(out.exists())
    def test_personal_checkpoint_rejected(self):
        self.api.releases['training-checkpoint']={'assets':[{'name':'checkpoint.json','data':json.dumps({'cloudStateSchema':1,'ledger':{}}).encode()}]}
        with self.assertRaises(RuntimeError):pipeline.restore_checkpoint(self.api,None,self.folder/'restored.json')
    def test_secret_redirection_strips_auth(self):
        from urllib.request import Request
        request=Request('https://api.github.com/assets',headers={'Authorization':'Bearer test-only'})
        redirected=pipeline.SafeRedirect().redirect_request(request,None,302,'',{},'https://release-assets.githubusercontent.com/file')
        self.assertIsNone(redirected.get_header('Authorization'))
        with self.assertRaises(RuntimeError):pipeline.SafeRedirect().redirect_request(request,None,302,'',{},'https://evil.test/file')
    def test_private_training_cannot_publish_into_same_repository(self):
        env={'GITHUB_REPOSITORY':'lins58308/oilcast-models','GITHUB_RUN_ID':'123','GITHUB_RUN_ATTEMPT':'1',
             'MODEL_PUBLIC_REPOSITORY':'lins58308/oilcast-models','MODEL_PUBLISH_TOKEN':'fake-test-token',
             'MODEL_SIGNING_KEY_B64':'fake-test-key'}
        with patch.dict('os.environ',env,clear=True):
            with self.assertRaisesRegex(RuntimeError,'must be separate'):pipeline.main()
    def test_private_training_needs_scoped_publisher_token(self):
        env={'GITHUB_REPOSITORY':'lins58308/oilcast-training-private','GITHUB_RUN_ID':'123','GITHUB_RUN_ATTEMPT':'1',
             'MODEL_PUBLIC_REPOSITORY':'lins58308/oilcast-models','MODEL_SIGNING_KEY_B64':'fake-test-key'}
        with patch.dict('os.environ',env,clear=True):
            with self.assertRaisesRegex(RuntimeError,'MODEL_PUBLISH_TOKEN'):pipeline.main()

if __name__=='__main__':unittest.main()
