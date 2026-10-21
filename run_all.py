"""python run_all.py --input inputs --out results_full
Inspect all supplied sources, audit road coordinates, compute endpoint candidates.
No geocoding and no assumed facility CRS. Uses analyze.py for source inspection.
"""
import argparse, subprocess, sys, json, html, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from analyze import save, distances, valid

def run(root,out):
    subprocess.run([sys.executable,str(Path(__file__).with_name('analyze.py')),'--input',str(root),'--out',str(out)],check=True)
    roads=[]; eps=[]; schemas=[]
    for p in sorted(root.glob('*.csv')):
        for enc in ['utf-8-sig','cp949','utf-16']:
            try: d=pd.read_csv(p,encoding=enc); break
            except UnicodeError: continue
        schemas.append({'file':p.name,'encoding':enc,'columns_json':json.dumps(list(d.columns),ensure_ascii=False)})
        cols=['기점위도','기점경도','종점위도','종점경도'] if '기점위도' in d else ['도로구간-기점(위도)','도로구간-기점(경도)','도로구간-종점(위도)','도로구간-종점(경도)']
        assert all(c in d for c in cols)
        for i,r in d.iterrows():
            vs=[pd.to_numeric(r[c],errors='coerce') for c in cols]; a,b,c,e=vs
            both=valid(a,b) and valid(c,e)
            chord=float(distances(a,b,[c],[e])[0]) if both else np.nan
            length=pd.to_numeric(r['총길이(km)'],errors='coerce')
            coincident=bool(both and chord<1 and length>0.01)
            impossible=bool(both and pd.notna(length) and length>0 and chord>length*1000*1.1+20)
            rid=f'{p.name}:{i+2}'
            roads.append(dict(road_id=rid,source_file=p.name,excel_row=i+2,district=r['시군구명'],route=r['노선명'],reported_length_km=length,data_date=r['데이터기준일자'],endpoint_chord_m=chord,coincident_endpoints_review=coincident,chord_exceeds_length_review=impossible,both_coordinates_valid=both))
            for role,lat,lon in [('start',a,b),('end',c,e)]:
                ok=valid(lat,lon)
                eps.append(dict(endpoint_id=rid+':'+role,road_id=rid,source_file=p.name,excel_row=i+2,district=r['시군구명'],route=r['노선명'],role=role,lat=lat,lon=lon,coordinate_valid=ok,strict_eligible=ok and not coincident and not impossible,coordinate_status='missing' if pd.isna(lat) or pd.isna(lon) else 'in_seoul_range' if ok else 'out_of_range',hotspot_50m_status='pending_precise_geocoding',facility_50m_status='pending_verified_facility_crs'))
    rd=pd.DataFrame(roads); ep=pd.DataFrame(eps); save(rd,out/'road_audit.csv'); save(ep,out/'endpoint_audit.csv'); save(pd.DataFrame(schemas),out/'road_column_mapping.csv')
    allmetrics=[]; pairs=[]
    for mode,flag in [('reported','coordinate_valid'),('strict','strict_eligible')]:
        subset=ep[ep[flag]]
        for _,r in subset.iterrows():
            ts=subset[subset.road_id!=r.road_id]; ds=distances(r.lat,r.lon,ts.lat,ts.lon)
            within=ts[ds<=50]
            allmetrics.append(dict(endpoint_id=r.endpoint_id,district=r.district,route=r.route,mode=mode,nearest_other_endpoint_m=float(ds.min()) if len(ds) else np.nan,other_endpoint_count_50m=len(within),other_road_record_count_50m=within.road_id.nunique(),no_other_endpoint_within_50m=not len(within),boundary_review_49_8_to_50_2m=bool(((ds>=49.8)&(ds<=50.2)).any())))
            for (_,t),dist in zip(ts.iterrows(),ds):
                if dist<=50 and r.endpoint_id<t.endpoint_id:
                    pairs.append(dict(mode=mode,endpoint_a=r.endpoint_id,endpoint_b=t.endpoint_id,distance_m=float(dist),cross_district=r.district!=t.district))
    met=pd.DataFrame(allmetrics); save(met,out/'endpoint_connectivity.csv'); save(pd.DataFrame(pairs),out/'endpoint_pairs_50m.csv')
    summaries=[]
    for district,g in rd.groupby('district'):
        e=ep[ep.district==district]; m=met[(met.district==district)&(met['mode']=='strict')]
        summaries.append(dict(district=district,road_records=len(g),reported_length_km=g.reported_length_km.sum(),endpoint_slots=len(e),valid_endpoint_coordinates=int(e.coordinate_valid.sum()),strict_eligible_endpoints=int(e.strict_eligible.sum()),strict_endpoints_with_candidate=int((~m.no_other_endpoint_within_50m).sum()),strict_endpoints_without_candidate=int(m.no_other_endpoint_within_50m.sum()),coincident_road_records=int(g.coincident_endpoints_review.sum()),length_review_road_records=int(g.chord_exceeds_length_review.sum()),dates=';'.join(sorted(g.data_date.dropna().astype(str).unique()))))
    sm=pd.DataFrame(summaries)
    sm.loc[sm.strict_eligible_endpoints==0,['strict_endpoints_with_candidate','strict_endpoints_without_candidate']]=np.nan
    save(sm,out/'district_road_summary.csv')
    dup=ep[ep.coordinate_valid & ep.duplicated(['lat','lon'],keep=False)]; save(dup,out/'coincident_coordinate_review.csv')
    features=[]; circles=[]; good=ep[ep.coordinate_valid]; lat0=good.lat.mean(); scale=np.cos(np.radians(lat0)); xx=good.lon*scale; yy=good.lat; k=min(900/(xx.max()-xx.min()),620/(yy.max()-yy.min()))
    counts=met[met['mode']=='strict'].set_index('endpoint_id').other_road_record_count_50m.to_dict()
    for _,r in good.iterrows():
        pr={'endpoint_id':r.endpoint_id,'district':r.district,'route':r.route,'strict_eligible':bool(r.strict_eligible),'other_road_records_50m':counts.get(r.endpoint_id),'hotspot_status':'pending_precise_geocoding'}
        features.append({'type':'Feature','geometry':{'type':'Point','coordinates':[r.lon,r.lat]},'properties':pr})
        color='#888888' if not r.strict_eligible else '#177f71' if counts[r.endpoint_id]>0 else '#ce5a20'
        circles.append(f'<circle cx="{45+(r.lon*scale-xx.min())*k:.2f}" cy="{690-(r.lat-yy.min())*k:.2f}" r="3" fill="{color}" opacity="0.75"><title>{html.escape(r.endpoint_id+" "+str(r.route))}</title></circle>')
    (out/'endpoints.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features},ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (out/'endpoint_map.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>자전거도로 끝점 분포</title><style>body{font-family:Arial,sans-serif;max-width:1100px;margin:24px auto}svg{border:1px solid #bbb;background:#fafafa}circle:hover{r:7}</style><h1>자전거도로 기점·종점 분포</h1><p>주황: 50m 이내 다른 도로 끝점 후보 없음 · 초록: 후보 있음 · 회색: 품질검토로 엄격 분석 제외</p><p>위·경도 기반 위치도. 배경지도·실제 노선·사고 좌표는 표시하지 않음. 범위 내 좌표만 표시. 미제공 도로와의 연결은 알 수 없음. 점에 마우스를 올리면 원본 행과 노선명 표시.</p><svg viewBox="0 0 1000 750" role="img" aria-label="자전거도로 끝점 위치도">'+''.join(circles)+'</svg></html>',encoding='utf-8')
    status=json.loads((out/'status.json').read_text(encoding='utf-8')); status.update(road_records=len(rd),endpoint_slots=len(ep),valid_endpoint_coordinates=int(ep.coordinate_valid.sum()),strict_eligible_endpoints=int(ep.strict_eligible.sum()),endpoint_analysis='computed_partial_coverage_8_districts',facility_proximity='pending_verified_facility_crs',hotspot_overlap='pending_precise_geocoding')
    (out/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); run(Path(a.input),Path(a.out))
