#!/usr/bin/env python3
"""Persist public training data, train on CPU, verify uploads, then publish signed models."""
import base64,datetime,hashlib,json,os,pathlib,re,subprocess,time,urllib.error,urllib.parse,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
MAX_STATE=32*1024*1024

class ApiError(RuntimeError):
    def __init__(self,status):self.status=status;super().__init__('GitHub operation failed (HTTP '+str(status)+')')

class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,message,headers,newurl):
        host=urllib.parse.urlsplit(newurl).hostname
        if host not in {'api.github.com','uploads.github.com','release-assets.githubusercontent.com','objects.githubusercontent.com'}:raise RuntimeError('Untrusted GitHub transfer redirect')
        redirected=super().redirect_request(request,fp,code,message,headers,newurl)
        if host!='api.github.com':redirected.remove_header('Authorization')
        return redirected

class GitHub:
    def __init__(self,repository,token):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repository) or not token:raise RuntimeError('Repository or workflow token is missing')
        self.repository=repository;self.token=token;self.base='https://api.github.com/repos/'+repository
        self.opener=urllib.request.build_opener(SafeRedirect())
    def request(self,path,method='GET',value=None,binary=None,raw=False,limit=MAX_STATE):
        url=path if path.startswith('https://') else self.base+path
        host=urllib.parse.urlsplit(url).hostname
        if host not in {'api.github.com','uploads.github.com'}:raise RuntimeError('Untrusted API endpoint')
        headers={'Accept':'application/octet-stream' if raw else 'application/vnd.github+json','User-Agent':'OilcastCloud/1','X-GitHub-Api-Version':'2026-03-10','Authorization':'Bearer '+self.token}
        data=binary
        if value is not None:data=json.dumps(value).encode();headers['Content-Type']='application/json'
        elif binary is not None:headers['Content-Type']='application/json'
        request=urllib.request.Request(url,data=data,headers=headers,method=method)
        try:
            with self.opener.open(request,timeout=45) as response:
                body=response.read(limit+1)
                if len(body)>limit:raise RuntimeError('GitHub response exceeds size limit')
        except urllib.error.HTTPError as error:raise ApiError(error.code) from None
        if raw:return body
        return json.loads(body) if body else None
    def release(self,tag):
        try:return self.request('/releases/tags/'+urllib.parse.quote(tag,safe=''))
        except ApiError as error:
            if error.status==404:return None
            raise
    def latest(self):
        try:return self.request('/releases/latest')
        except ApiError as error:
            if error.status==404:return None
            raise
    def asset(self,release,name):
        matches=[a for a in release.get('assets',[]) if a['name']==name]
        if len(matches)!=1:raise RuntimeError('Training/model asset missing or duplicated')
        return self.request(matches[0]['url'],raw=True)
    def upload(self,release,name,data,replace=False):
        if replace:
            for asset in release.get('assets',[]):
                if asset['name']==name:self.request('/releases/assets/'+str(asset['id']),method='DELETE')
        url=release['upload_url'].split('{',1)[0]+'?'+urllib.parse.urlencode({'name':name})
        result=self.request(url,method='POST',binary=data)
        if result.get('size')!=len(data):raise RuntimeError('Uploaded asset size differs')
        readback=self.request(result['url'],raw=True)
        if hashlib.sha256(readback).digest()!=hashlib.sha256(data).digest():raise RuntimeError('Uploaded asset checksum differs')
        return result
    def status(self,value,name='status.json'):
        if name not in {'status.json','status-v7.json'}:raise RuntimeError('Invalid status channel')
        path='/contents/'+name;sha=None
        try:sha=self.request(path)['sha']
        except ApiError as error:
            if error.status!=404:raise
        request={'message':'Update public cloud training status','content':base64.b64encode(json.dumps(value,separators=(',',':')).encode()).decode()}
        if sha:request['sha']=sha
        self.request(path,method='PUT',value=request)

def restore_checkpoint(api,latest,output,checkpoint_tag='training-checkpoint'):
    checkpoint=api.release(checkpoint_tag)
    if checkpoint:
        data=api.asset(checkpoint,'checkpoint.json');state=json.loads(data)
        if state.get('cloudStateSchema')!=1 or any(k in state for k in ['ledger','authorization','bills']):raise RuntimeError('Invalid public training checkpoint')
        output.write_bytes(data)
    elif latest:
        # Never silently reset the revision/history after a checkpoint upload failure.
        data=api.asset(latest,'model.json');payload=json.loads(data)
        if 'multisource' not in payload or any(k in payload for k in ['ledger','authorization','issued']):raise RuntimeError('Latest public model cannot restore training data')
        output.write_text(json.dumps({'cloudStateSchema':1,'rows':payload['rows'],'multisource':payload['multisource'],'model':payload['model'],'cloudAccuracy':payload.get('cloudAccuracy')}))
    else:return
    # A model may have been published immediately before a checkpoint update failed.
    if latest:
        payload=json.loads(api.asset(latest,'model.json'));old=json.loads(output.read_text())
        if payload['model']['trainedAt']>old['model']['trainedAt']:
            output.write_text(json.dumps({'cloudStateSchema':1,'rows':payload['rows'],'multisource':payload['multisource'],'model':payload['model'],'cloudAccuracy':payload.get('cloudAccuracy')}))

def publish(api,tag,folder,test_channel=False):
    envelope=json.loads((folder/'envelope.json').read_text());manifest=json.loads(base64.b64decode(envelope['manifestBase64']))
    if manifest.get('demo') or 'payloadUrl' not in manifest:raise RuntimeError('Demo/unsigned output cannot be published')
    model=(folder/'model.json').read_bytes()
    if hashlib.sha256(model).hexdigest()!=manifest['payloadSha256'] or len(model)!=manifest['payloadBytes']:raise RuntimeError('Model package checksum mismatch')
    if not manifest['payloadUrl'].startswith('https://github.com/'+api.repository+'/releases/download/'+tag+'/'):raise RuntimeError('Immutable model URL points to another repository/release')
    if api.release(tag):raise RuntimeError('Immutable model release already exists; inspect it before retrying')
    draft=api.request('/releases',method='POST',value={'tag_name':tag,'name':('Oilcast neural v7 TEST ' if test_channel else 'Oilcast model ')+manifest['trainedAt'],'body':'Public oil-price model. Forecasts are estimates; official CPC announcements take precedence.','draft':True,'prerelease':test_channel,'make_latest':'false'})
    api.upload(draft,'model.json',model)
    api.upload(draft,'envelope.json',(folder/'envelope.json').read_bytes())
    # Both immutable assets have been read back before becoming the public latest model.
    release=api.request('/releases/'+str(draft['id']),method='PATCH',value={'draft':False,'make_latest':'false' if test_channel else 'true'})
    checkpoint_tag='training-checkpoint-v7' if test_channel else 'training-checkpoint'
    checkpoint=api.release(checkpoint_tag)
    if not checkpoint:checkpoint=api.request('/releases',method='POST',value={'tag_name':checkpoint_tag,'name':'Public training checkpoint'+(' (v7 test)' if test_channel else ''),'draft':False,'prerelease':True,'make_latest':'false'})
    api.upload(checkpoint,'checkpoint.json',(folder/'checkpoint.json').read_bytes(),replace=True)
    return release

def main():
    repo=os.environ.get('GITHUB_REPOSITORY','');run_id=os.environ.get('GITHUB_RUN_ID','');attempt=os.environ.get('GITHUB_RUN_ATTEMPT','1')
    public_repo=os.environ.get('MODEL_PUBLIC_REPOSITORY','')
    publish_token=os.environ.get('MODEL_PUBLISH_TOKEN','')
    test_channel=os.environ.get('MODEL_TEST_CHANNEL','false').lower()=='true'
    if test_channel:
        direct_branch=(os.environ.get('GITHUB_REF_NAME')=='feature/neural-v7-backtesting' and os.environ.get('GITHUB_EVENT_NAME') in {'push','workflow_dispatch'})
        trusted_schedule=(os.environ.get('GITHUB_REF_NAME')=='main' and os.environ.get('GITHUB_WORKFLOW')=='Maintain OilCast NN v7 public test feed' and os.environ.get('GITHUB_EVENT_NAME') in {'schedule','workflow_dispatch'})
        if not (direct_branch or trusted_schedule):raise RuntimeError('v7 test publisher can only run on its isolated branch or trusted maintenance workflow')
    if not re.fullmatch(r'[0-9]+',run_id) or not re.fullmatch(r'[0-9]+',attempt):raise RuntimeError('This publisher runs only inside a configured GitHub workflow')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',public_repo):raise RuntimeError('MODEL_PUBLIC_REPOSITORY is missing or invalid')
    if public_repo==repo:raise RuntimeError('Training and public-download repositories must be separate')
    if not publish_token:raise RuntimeError('MODEL_PUBLISH_TOKEN is not configured')
    if not os.environ.get('MODEL_SIGNING_KEY_B64'):raise RuntimeError('MODEL_SIGNING_KEY_B64 is not configured')
    api=GitHub(public_repo,publish_token);tag='model-'+run_id+'-'+attempt;out=ROOT/'output';out.mkdir(exist_ok=True)
    status_name='status-v7.json' if test_channel else 'status.json'
    checkpoint_tag='training-checkpoint-v7' if test_channel else 'training-checkpoint'
    # The existing release/latest remains the v6 channel; v7 has an independently
    # versioned status pointer and never sets GitHub's latest release.
    previous_status={}
    if test_channel:
        try:
            entry=api.request('/contents/status-v7.json')
            previous_status=json.loads(base64.b64decode(entry['content']))
        except ApiError as error:
            if error.status!=404:raise
        model_tag=previous_status.get('modelTag','')
        if model_tag and not re.fullmatch(r'model-[0-9]+-[0-9]+',model_tag):raise RuntimeError('Invalid existing v7 model pointer')
    else:model_tag=''
    # A public status file must not expose the private Actions run URL.
    run_url='https://github.com/'+public_repo+'/releases/latest'
    def report(phase,percent,outcome='running'):
        status={'schema':1,'outcome':outcome,'phase':phase,'percent':percent,'updatedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runUrl':run_url}
        if test_channel:status.update({'channel':'neural-v7','modelTag':model_tag})
        api.status(status,status_name)
    process=None
    try:
        report('prepare',0)
        latest=api.release(model_tag) if test_channel and model_tag else api.latest()
        restore_checkpoint(api,latest,ROOT/'checkpoint.json',checkpoint_tag)
        url='https://github.com/'+public_repo+'/releases/download/'+tag+'/model.json'
        process=subprocess.Popen(['python3',str(ROOT/'run_training.py'),'--live','--checkpoint',str(ROOT/'checkpoint.json'),'--output',str(out),'--model-url',url],cwd=ROOT)
        last_phase=None;last_report=0
        while process.poll() is None:
            try:status=json.loads((out/'training-status.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):status=None
            if status and time.monotonic()-last_report>=10:
                phase=status['phase'];percent=min(95,status['percent'])
                if phase!=last_phase or time.monotonic()-last_report>=30:report(phase,percent);last_report=time.monotonic();last_phase=phase
            time.sleep(2)
        if process.returncode!=0:raise RuntimeError('Training failed; previous public model stays available')
        cp=str(ROOT/'build/classes')+os.pathsep+str(ROOT/'build/json.jar')
        subprocess.run(['java','-cp',cp,'VerifyOutput',str(ROOT),str(out)],check=True)
        report('publish',96);publish(api,tag,out,test_channel)
        subprocess.run(['java','-cp',cp,'VerifyPublicDownload',str(ROOT),public_repo,str(out),tag if test_channel else 'latest'],check=True)
        if test_channel:model_tag=tag
        report('complete',100,'complete')
        print('Signed cloud model published:',tag)
    except Exception:
        if process and process.poll() is None:process.terminate()
        try:report('failed',0,'failed')
        except Exception:pass
        raise

if __name__=='__main__':main()
