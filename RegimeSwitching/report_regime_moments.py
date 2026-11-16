"""Regenerate best-model pages and moment outputs without fitting or rewriting estimates."""
import hashlib
import pandas as pd
import estimate_regime_switching_models as rs


def main():
    for frequency, folder in [('Quarterly','results'),('Monthly','results_monthly')]:
        out = rs.RS_DIR/folder
        paths = [out/'regime_switching_results.csv',out/'regime_switching_parameters.csv',
                 rs.ROOT/'DataSummary'/f'Aggregate_CPI_inflation_{frequency}.pkl']
        before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        summary=pd.read_csv(paths[0],float_precision='round_trip')
        params=pd.read_csv(paths[1],float_precision='round_trip')
        data=rs.load_effective_sample(paths[2])
        eligible=summary[summary.checks_passed & summary.stability_verified]
        row=eligible.loc[eligible.aic.idxmin()]
        rs._check_saved_sample(data,row)
        subset=params[params.model_label==row.model_label]
        result=rs._reconstruct_result_from_saved_params(data,row,subset)
        if abs(result.llf-row.llf)>1e-6:
            raise ValueError('Selected likelihood is not reproducible')
        text=rs.build_best_model_markdown(row,subset,result,data,out)
        (out/'regime_switching_best_model.md').write_text(rs.style_monthly_report(text, paths[2])+'\n')
        assert all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in before.items())
        print(f'{frequency}: {row.model_label}; N={len(data)}; saved estimates unchanged.')
        print(pd.read_csv(out/'best_model_moment_summary.csv').to_string(index=False))


if __name__=='__main__':
    main()
