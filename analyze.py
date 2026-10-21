"""Reproduce inspection and partial analysis. Python 3.11+, pandas, numpy, openpyxl.
Usage: python analyze.py --input inputs --out results [--spatial spatial.json]
Spatial JSON explicitly maps reviewed columns; see README.md. No geocoding occurs.
"""
import argparse, hashlib, json, math, zipfile, html
from pathlib import Path
import numpy as np
import pandas as pd

def save(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig')

def distances(lat, lon, lats, lons):
    a,b,c,d = map(np.radians, (lat,lon,np.asarray(lats,float),np.asarray(lons,float)))
    h=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 6371008.8*2*np.arcsin(np.sqrt(np.clip(h,0,1)))

def valid(lat,lon):
    return np.isfinite(lat) and np.isfinite(lon) and 37.3<=lat<=37.8 and 126.7<=lon<=127.3

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--out',required=True); ap.add_argument('--spatial')
    args=ap.parse_args(); root=Path(args.input); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    inventory=[]; hotspots=[]; facilities=[]; tables={}; excluded=[]
    files=sorted(p for p in root.rglob('*') if p.suffix.lower() in ['.csv','.xls','.xlsx'])
    for p in files:
        raw=p.read_bytes(); fmt='OOXML' if zipfile.is_zipfile(p) else 'OLE' if raw[:4]==bytes.fromhex('d0cf11e0') else 'text'
        key=p.relative_to(root).as_posix()
        if p.suffix.lower()=='.csv':
            for enc in ['utf-8-sig','cp949','utf-16']:
                try: sheets={'CSV':pd.read_csv(p,encoding=enc)}; break
                except UnicodeError: continue
            else: raise ValueError(f'Unknown encoding: {p}')
        else:
            sheets=pd.read_excel(p,engine='openpyxl' if fmt=='OOXML' else 'xlrd',sheet_name=None,header=None)
        for sheet,d in sheets.items():
            inventory.append(dict(file=key,sheet=sheet,actual_format=fmt,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),raw_rows=len(d),raw_columns=len(d.columns),first_rows_json=d.head(4).to_json(orient='values',force_ascii=False)))
            if '교통약자다발지점' in p.name:
                assert d.shape[1]==10, f'Review changed hotspot layout: {p}'
                assert '발생건수' in str(d.iloc[1,3]), f'Review hotspot header: {p}'
                for i,r in d.iloc[3:].iterrows():
                    if pd.isna(r.iloc[2]): continue
                    if pd.isna(r.iloc[1]) or not str(r.iloc[1]).endswith('구'):
                        excluded.append(dict(source_file=key,excel_row=i+1,reason='summary_or_non_district_row',raw_json=r.to_json(force_ascii=False)))
                        continue
                    hotspots.append(dict(source_file=key,source_sheet=sheet,excel_row=i+1,city=r.iloc[0],district=r.iloc[1],location=r.iloc[2],occurrences=pd.to_numeric(r.iloc[3],errors='coerce'),deaths=pd.to_numeric(r.iloc[4],errors='coerce'),MOTOCY_CNT=r.iloc[6],CYCLE_CNT=r.iloc[7],WALK_CNT=r.iloc[8],DRK=r.iloc[9],geocoding_status='pending_precise_geocoding'))
            elif '횡단보도' in p.name:
                d.columns=d.iloc[0].astype(str); d=d.iloc[1:].copy()
                d['source_file']=key; d['source_sheet']=sheet; d['excel_row']=np.arange(2,len(d)+2)
                facilities.append(d)
            tables[(key,sheet)]=d
    save(pd.DataFrame(inventory),out/'file_inventory.csv')
    save(pd.DataFrame(excluded),out/'excluded_hotspot_rows.csv')
    hs=pd.DataFrame(hotspots); fs=pd.concat(facilities,ignore_index=True) if facilities else pd.DataFrame()
    if len(hs):
        save(hs,out/'hotspots_pending_geocoding.csv')
        summary=hs.groupby('district',dropna=False).agg(hotspot_rows=('location','size'),reported_occurrences=('occurrences','sum'),reported_deaths=('deaths','sum')).reset_index()
        save(summary,out/'district_hotspot_summary.csv')
    if len(fs):
        save(fs,out/'facilities_raw_coordinates.csv')
        save(fs.groupby('자치구').size().rename('facility_rows').reset_index(),out/'district_facility_summary.csv')
        x=pd.to_numeric(fs['X좌표'],errors='coerce'); y=pd.to_numeric(fs['Y좌표'],errors='coerce'); ok=x.notna()&y.notna()
        # Offline source-coordinate plot, deliberately not a geographic basemap.
        xx=x[ok]; yy=y[ok]; w=1000; h=760
        pts=''.join(f'<circle cx="{40+(a-xx.min())/(xx.max()-xx.min())*920:.1f}" cy="{710-(b-yy.min())/(yy.max()-yy.min())*630:.1f}" r="1.2" fill="#146c94"><title>{html.escape(str(n))}</title></circle>' for a,b,n in zip(xx,yy,fs.loc[ok,'교차로명']))
        (out/'facility_coordinate_plot.html').write_text('<!doctype html><meta charset="utf-8"><title>시설 원본좌표 분포</title><h1>교차로·횡단보도 시설 원본 X/Y 분포</h1><p>좌표계 미확인. 지리 지도 또는 끝점 분석 결과가 아님. 각 점은 시설 행 1개.</p><svg viewBox="0 0 1000 760" role="img" aria-label="원본 X Y 시설 분포">'+pts+'</svg>',encoding='utf-8')
    status={'input_files':len(files),'road_csv_files':sum(p.suffix.lower()=='.csv' for p in files),'hotspot_files':len(set(hs.source_file)) if len(hs) else 0,'hotspot_rows':len(hs),'facility_rows':len(fs),'endpoint_analysis':'pending_road_csv','hotspot_overlap':'pending_precise_geocoding','facility_proximity':'pending_road_csv_and_verified_crs'}
    if args.spatial:
        cfg=json.loads(Path(args.spatial).read_text(encoding='utf-8'))
        assert cfg['coordinate_crs']=='EPSG:4326' and cfg['coordinate_provenance'], 'Provide verified WGS84 coordinates and provenance'
        endpoints=[]; targets=[]
        for item in cfg['datasets']:
            d=pd.read_csv(root/item['file'],encoding=item.get('encoding','utf-8-sig'))
            assert item['provenance'], 'Each dataset requires coordinate provenance'
            for i,r in d.iterrows():
                roles=['start','end'] if item['kind']=='road' else ['point']
                for role in roles:
                    cols=item[role]; lat=pd.to_numeric(r[cols[0]],errors='coerce'); lon=pd.to_numeric(r[cols[1]],errors='coerce')
                    rec=dict(id=f"{item['file']}:{i+2}:{role}",road_id=f"{item['file']}:{i+2}",lat=lat,lon=lon,kind=item['kind'],valid=valid(lat,lon))
                    (endpoints if item['kind']=='road' else targets).append(rec)
        ep=pd.DataFrame(endpoints); tg=pd.DataFrame(targets)
        save(ep,out/'endpoint_coordinate_audit.csv'); pairs=[]; metrics=[]
        for _,e in ep[ep.valid].iterrows():
            m={'endpoint_id':e.id}
            for kind in ['road','hotspot','intersection','crosswalk']:
                candidates=ep[(ep.valid)&(ep.road_id!=e.road_id)] if kind=='road' else (tg[(tg.valid)&(tg.kind==kind)] if len(tg) else pd.DataFrame())
                # Empty/absent targets are unknown coverage, not a zero overlap.
                m[kind+'_nearest_m']=None; m[kind+'_within_50m']=None
                if len(candidates):
                    ds=distances(e.lat,e.lon,candidates.lat,candidates.lon); m[kind+'_nearest_m']=float(ds.min()); m[kind+'_within_50m']=int((ds<=50).sum())
                    for (_,t),dist in zip(candidates.iterrows(),ds):
                        if dist<=50: pairs.append(dict(endpoint_id=e.id,target_id=t.id,kind=kind,distance_m=float(dist)))
            metrics.append(m)
        save(pd.DataFrame(metrics),out/'endpoint_metrics.csv'); save(pd.DataFrame(pairs,columns=['endpoint_id','target_id','kind','distance_m']),out/'overlap_pairs_50m.csv')
        status['endpoint_analysis']='computed_for_verified_coordinates_only'
    (out/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False))

if __name__=='__main__': main()
