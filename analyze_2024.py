"""python analyze_2024.py --input inputs --out results
Read supplied originals, rebuild road screening, reconcile official 2024 hotspot
responses, and compare hotspot center proximity at 50/100/150 m. No geocoding.
"""
import argparse,json,subprocess,sys,hashlib,html
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont

def save(d,p): d.to_csv(p,index=False,encoding='utf-8-sig')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='inputs'); ap.add_argument('--out',default='results'); a=ap.parse_args(); root=Path(a.input); out=Path(a.out)
    subprocess.run([sys.executable,str(Path(__file__).with_name('run_all.py')),'--input',str(root),'--out',str(out)],check=True)
    raw=[]; audits=[]
    for p in sorted(root.glob('*.json')):
        records=json.loads(p.read_text(encoding='utf-8-sig')); assert isinstance(records,list)
        for r in records:
            assert str(r['stdYear'])=='2024' and str(r['extractCd'])=='521' and r['extractRadius']==50
            r={**r,'source_file':p.name}; raw.append(r)
        audits.append({'file':p.name,'rows':len(records),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    h=pd.DataFrame(raw); assert len(h)==308 and h.posNo.nunique()==308 and h.jijaceNm.nunique()==11
    assert h.wgsX.between(126.7,127.3).all() and h.wgsY.between(37.3,37.8).all()
    old=pd.read_csv(out/'hotspots_pending_geocoding.csv'); reconciliation=[]
    for district,g in h.groupby('jijaceNm'):
        x=old[old.district==district]
        matched=Counter(zip(g.posNa,g.accCnt,g.deathCnt))==Counter(zip(x.location,x.occurrences,x.deaths))
        assert matched, f'XLS mismatch: {district}'
        reconciliation.append({'district':district,'json_hotspots':len(g),'xls_hotspot_rows':len(x),'name_count_death_multiset_match':matched,'reported_cluster_occurrences':int(g.accCnt.sum()),'reported_deaths':int(g.deathCnt.sum())})
    save(pd.DataFrame(reconciliation),out/'hotspot_xls_reconciliation.csv'); save(pd.DataFrame(audits),out/'coordinate_source_manifest.csv')
    clean=h[['posNo','stdYear','jijaceNm','posNa','poiNa','wgsX','wgsY','extractRadius','accCnt','deathCnt','source_file']].rename(columns={'posNo':'hotspot_id','stdYear':'reference_year','jijaceNm':'district','posNa':'hotspot_name','poiNa':'source_address_unverified','wgsX':'longitude','wgsY':'latitude','extractRadius':'cluster_radius_m','accCnt':'reported_cluster_occurrences','deathCnt':'reported_deaths'})
    save(clean,out/'official_hotspots_2024.csv')
    # Supersede the intermediate label emitted by the original reader.
    old['geocoding_status']='official_hotspot_centers_available_see_official_hotspots_2024'; save(old,out/'hotspots_pending_geocoding.csv')
    epa=pd.read_csv(out/'endpoint_audit.csv'); epa['hotspot_50m_status']='computed_2024_hotspot_center_proximity'; save(epa,out/'endpoint_audit.csv')
    metrics=[]; pairs=[]; sums=[]; district_sums=[]; matrix_by_mode={}
    for mode,flag in [('reported','coordinate_valid'),('strict','strict_eligible')]:
        ep=epa[epa[flag]].reset_index(drop=True)
        la=np.radians(clean.latitude.to_numpy())[:,None]; lo=np.radians(clean.longitude.to_numpy())[:,None]
        ela=np.radians(ep.lat.to_numpy())[None,:]; elo=np.radians(ep.lon.to_numpy())[None,:]
        hv=np.sin((ela-la)/2)**2+np.cos(ela)*np.cos(la)*np.sin((elo-lo)/2)**2
        dist=2*6371008.8*np.arcsin(np.sqrt(np.clip(hv,0,1)))
        independent=6371008.8*np.arccos(np.clip(np.sin(la)*np.sin(ela)+np.cos(la)*np.cos(ela)*np.cos(elo-lo),-1,1))
        assert np.allclose(dist,independent,atol=.15)
        matrix_by_mode[mode]=(ep,dist)
        for j,e in enumerate(ep.itertuples()):
            k=int(dist[:,j].argmin()); near=clean.iloc[k]
            row={'mode':mode,'endpoint_id':e.endpoint_id,'district':e.district,'route':e.route,'nearest_hotspot_id':near.hotspot_id,'nearest_hotspot_name':near.hotspot_name,'nearest_hotspot_district':near.district,'nearest_hotspot_center_m':float(dist[k,j])}
            for rad in [50,100,150]: row[f'hotspot_centers_{rad}m']=int((dist[:,j]<=rad).sum())
            assert row['hotspot_centers_50m']<=row['hotspot_centers_100m']<=row['hotspot_centers_150m']; metrics.append(row)
        for k,j in np.argwhere(dist<=150):
            e=ep.iloc[j]; r=clean.iloc[k]; dd=float(dist[k,j])
            pairs.append({'mode':mode,'endpoint_id':e.endpoint_id,'endpoint_district':e.district,'route':e.route,'hotspot_id':r.hotspot_id,'hotspot_district':r.district,'hotspot_name':r.hotspot_name,'distance_m':dd,'within_50m':dd<=50,'within_100m':dd<=100,'within_150m':True,'cross_district':e.district!=r.district})
        for rad in [50,100,150]:
            near=dist<=rad
            sums.append({'mode':mode,'radius_m':rad,'endpoint_denominator':len(ep),'hotspot_denominator':len(clean),'endpoints_near_hotspot':int(near.any(axis=0).sum()),'hotspots_near_endpoint':int(near.any(axis=1).sum()),'matching_pairs':int(near.sum()),'boundary_review_pairs_within_0_3m':int((abs(dist-rad)<=.3).sum())})
            for district in sorted(clean.district.unique()):
                em=(ep.district==district).to_numpy(); hm=(clean.district==district).to_numpy(); n=int(em.sum())
                # Cross-district proximity is allowed; both sides use all provided targets.
                district_sums.append({'mode':mode,'district':district,'radius_m':rad,'endpoint_denominator':n,'district_hotspots':int(hm.sum()),'endpoints_near_any_supplied_hotspot':int(near[:,em].any(axis=0).sum()) if n else None,'district_hotspots_near_any_supplied_endpoint':int(near[hm,:].any(axis=1).sum()),'endpoint_near_rate':float(near[:,em].any(axis=0).mean()) if n else None,'coverage_status':'partial_endpoint_coverage' if n else 'no_road_coordinates'})
    met=pd.DataFrame(metrics); pair=pd.DataFrame(pairs); sm=pd.DataFrame(sums)
    assert not pair.duplicated(['mode','endpoint_id','hotspot_id']).any()
    for r in sm.itertuples(): assert len(pair[(pair['mode']==r.mode)&(pair.distance_m<=r.radius_m)])==r.matching_pairs
    save(met,out/'endpoint_hotspot_metrics.csv'); save(pair,out/'endpoint_hotspot_pairs_150m.csv'); save(sm,out/'radius_comparison.csv'); save(pd.DataFrame(district_sums),out/'district_radius_comparison.csv')
    connectivity=pd.read_csv(out/'endpoint_connectivity.csv'); review=met[(met['mode']=='strict')&(met.hotspot_centers_100m>0)].merge(connectivity[connectivity['mode']=='strict'][['endpoint_id','no_other_endpoint_within_50m']],on='endpoint_id',validate='one_to_one')
    save(review[review.no_other_endpoint_within_50m],out/'review_endpoints_no_connection50m_near_hotspot100m.csv')
    features=[{'type':'Feature','geometry':{'type':'Point','coordinates':[r.longitude,r.latitude]},'properties':{'hotspot_id':r.hotspot_id,'district':r.district,'name':r.hotspot_name,'year':2024,'cluster_radius_m':50}} for r in clean.itertuples()]
    (out/'hotspots_2024.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features},ensure_ascii=False,allow_nan=False),encoding='utf-8')
    draw_maps(clean,matrix_by_mode['strict'][0],met[met['mode']=='strict'],out)
    status=json.loads((out/'status.json').read_text(encoding='utf-8')); status.update(reference_year=2024,accident_window='2022-01-01 through 2024-12-31',hotspot_rows=308,hotspot_overlap='computed_official_hotspot_centers_50_100_150m',facility_proximity='pending_verified_facility_crs'); (out/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    verification={'checks':'passed','official_hotspots':308,'unique_ids':308,'all_11_xls_districts_reconciled':True,'independent_distance_formula':'spherical cosine; maximum tolerance 0.15m','monotonic_radii':'passed','pair_counts':'reconciled','duplicate_pairs':0,'source_coordinates':'TMACS wgsX/wgsY; no geocoding','not_verified':['source survey accuracy/datum documentation','facility CRS','road geometry','browser interactivity'],'code_version':'2024-only full 11 district rebuild'}
    (out/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8'); print(sm.to_json(orient='records')); print('Review endpoints:',len(review[review.no_other_endpoint_within_50m]))

def draw_maps(h,ep,met,out):
    coords=pd.concat([pd.DataFrame({'lat':h.latitude,'lon':h.longitude}),ep[['lat','lon']]],ignore_index=True)
    cs=np.cos(np.radians(coords.lat.mean())); xmin=coords.lon.min()*cs; ymin=coords.lat.min(); dx=coords.lon.max()*cs-xmin; dy=coords.lat.max()-ymin; k=min(1050/dx,680/dy)
    def xy(lat,lon): return 95+(lon*cs-xmin)*k,900-(lat-ymin)*k
    im=Image.new('RGB',(1280,1020),'white'); dr=ImageDraw.Draw(im); font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20); title=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',28)
    dr.text((40,20),'2024년 자전거 사고다발지점과 도로 끝점',font=title,fill='#1e3344')
    dr.text((40,65),'원본 사고다발지점 308개(파랑) · 분석용 끝점 468개',font=font,fill='#333333')
    dr.text((40,100),'끝점: 100m 이내 사고다발지점 있음(주황) / 없음(초록)',font=font,fill='#333333')
    dr.text((40,135),'중심점 근접성 · 개별 사고 위치/실제 도로 선형/배경지도는 표시하지 않음',font=font,fill='#333333')
    dr.rectangle((90,205,1170,910),outline='#bbbbbb'); dots=[]
    for r in h.itertuples():
        x,y=xy(r.latitude,r.longitude); dr.ellipse((x-3,y-3,x+3,y+3),fill='#315ab5')
        dots.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#315ab5"><title>{html.escape(str(r.district)+" "+r.hotspot_name+" "+str(r.hotspot_id))}</title></circle>')
    data=met.set_index('endpoint_id')
    for e in ep.itertuples():
        x,y=xy(e.lat,e.lon); r=data.loc[e.endpoint_id]; color='#d45c1a' if r.hotspot_centers_100m>0 else '#198570'; dr.ellipse((x-3,y-3,x+3,y+3),fill=color)
        dots.append(f'<circle class="endpoint" data-distance="{r.nearest_hotspot_center_m:.8f}" cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"><title>{html.escape(e.district+" "+str(e.route)+" / "+e.endpoint_id)}; 최근접 중심 {r.nearest_hotspot_center_m:.1f}m</title></circle>')
    dr.text((40,940),'강서·노원·용산구 도로 끝점 좌표 없음. 다른 구도 일부 좌표·품질 제외.',font=font,fill='#333333'); dr.text((40,975),'사고집계: 2022~2024년 · 도로자료: 2025~2026년. 위험·인과관계로 해석 불가.',font=font,fill='#333333'); im.save(out/'map_2024_100m.png')
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>2024년 끝점–사고다발지점</title><style>body{font-family:Malgun Gothic,sans-serif;max-width:1200px;margin:24px auto}svg{width:100%;border:1px solid #ddd}circle:hover{r:7}</style><h1>2024년 자전거 사고다발지점과 도로 끝점</h1><p>파랑: 사고다발지점 308개. 끝점 468개: 주황은 선택 반경 안에 중심점 있음, 초록은 없음.</p><p>개별 사고 건수 분석이 아닙니다. 강서·노원·용산구 도로 끝점 좌표 없음. 실제 노선·배경지도는 생략.</p><label>분석 반경 <select id="radius"><option>50</option><option selected>100</option><option>150</option></select>m</label><svg viewBox="0 190 1280 750" role="img" aria-label="사고다발지점 중심과 도로 끝점 위치도">'+''.join(dots)+'</svg><script>document.getElementById("radius").addEventListener("change",function(){document.querySelectorAll(".endpoint").forEach(e=>{e.setAttribute("fill",Number(e.dataset.distance)<=Number(this.value)?"#d45c1a":"#198570")})});</script></html>'
    (out/'map_2024_interactive.html').write_text(page,encoding='utf-8')

if __name__=='__main__': main()
