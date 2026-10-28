"""Build the checked-in v3 summary from the collected validation artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/continuation_20261028'


def main():
    files = json.loads((REPORTS/'validation_v3_results.json').read_text(encoding='utf-8'))['files']
    progress = files['progress.json']
    assert progress['phase'] == 'complete' and not progress['test_evaluated']
    assert len(progress['stages']) == 17
    assert all(stage['returncode'] == 0 for stage in progress['stages'])
    candidate = progress['evaluation_sha256']
    models = {}
    for run in progress['declared_runs']:
        artifact = files[run+'.json']
        assert artifact['sources']['samples_validation'] == candidate
        assert not artifact['test_evaluated']
        models[run] = artifact['splits']['validation']
    baseline = files['baseline/metrics.json']
    assert baseline['sources']['samples_validation'] == candidate
    models.update(baseline['splits']['validation'])
    cal = files['ufm_full_v1_calibration.json']
    assert cal['sources']['samples_validation_sha256'] == candidate
    summary = dict(protocol='validation_v3', candidate_sha256=candidate,
        samples=40000, candidates_per_case=100, seed=42, test_evaluated=False,
        checkpoint_selection='historical validation v1',
        cold_macro_ndcg10={run: value['cold_macro_ndcg@10'] for run,value in models.items()},
        full_calibration=dict(temperature=cal['temperature'],fit_samples=cal['fit_samples'],
            audit_samples=cal['audit_samples'],
            before={key:cal['before']['overall'][key] for key in ['ece','nll','brier_multiclass']},
            after={key:cal['after']['overall'][key] for key in ['ece','nll','brier_multiclass']}),
        limitations=progress['limitations']+cal['limitations'])
    (REPORTS/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines = ['# Kết quả tiếp tục ngày 28/10/2026', '',
        'Nguồn: [toàn bộ kết quả v3](validation_v3_results.json), [tóm tắt máy đọc](summary.json), '
        '[audit validation v1](saved_validation_v1_audit.json).', '',
        '## Validation v3 đã hoàn tất', '',
        '17/17 bước thành công: popularity/TF-IDF, tám checkpoint UFM và tám lượt calibration. '
        'Mỗi mô hình dùng cùng 40.000 ca, 1 positive + 99 negative. Seed huấn luyện: 42. '
        'Checkpoint đã chọn từ validation v1; v3 là đánh giá lại, chưa phải holdout độc lập. '
        'Lượt này chỉ đọc validation, không đánh giá thêm test.', '',
        '`samples_validation.npz` SHA-256: `'+candidate+'`.', '',
        'Cold macro NDCG@10 (trung bình zero-shot, extreme-cold, cold):', '']
    for run,value in sorted(summary['cold_macro_ndcg10'].items(),key=lambda item:-item[1]):
        lines.append(f'- `{run}`: **{value:.6f}**.')
    lines += ['', 'Full UFM thấp hơn semantic-only và TF-IDF trên giao thức này. '
        'Kết quả chưa xác nhận lợi ích của gate/uncertainty đối với chỉ số chính. '
        'Một seed chưa đo được biến thiên giữa các lần huấn luyện; không coi bootstrap theo user là thay thế nhiều seed.', '',
        'TF-IDF dùng 10 history item, UFM dùng 20; đây là benchmark hệ thống. Các biến thể UFM dùng cùng CLIP và history.', '',
        '## Calibration thăm dò của full UFM', '',
        f"Fit {cal['fit_samples']:,} ca validation sớm; audit {cal['audit_samples']:,} ca muộn, giữ các timestamp bằng nhau cùng phía. "
        f"T = {cal['temperature']:.6f}; ECE {cal['before']['overall']['ece']:.6f} → {cal['after']['overall']['ece']:.6f}; "
        f"NLL {cal['before']['overall']['nll']:.6f} → {cal['after']['overall']['nll']:.6f}.", '',
        'Đây là confidence có điều kiện trên 100 ứng viên đã lấy mẫu, không phải xác suất mua. '
        'Checkpoint từng dùng validation để chọn; các phần theo thời gian có thể chung user. '
        'Temperature scaling không thay đổi thứ hạng. Các số calibration lịch sử v1 giữ riêng.', '',
        '![Reliability trên nửa validation audit](reliability_full_v3.png)', '',
        '## Bằng chứng lịch sử đã xác minh', '',
        'Audit v1 xác minh 10 lượt hoàn tất: full UFM, bảy ablation, SASRec và BERT4Rec. '
        'Các UFM ablation dùng cùng CLIP, code huấn luyện và ngân sách cấu hình. '
        'Tên run tuần tự có hậu tố v2 là phiên bản triển khai; predictions đã xác minh thuộc candidate v1. '
        'Chênh cold macro full − semantic-only trên v1 = −0,014931; paired user bootstrap 95% '
        '[−0,016918; −0,013001], 1.000 mẫu, 38.967 user. Không ghép metric v1 vào bảng v3.', '',
        'SASRec/BERT4Rec là bản thích nghi trong repo, chưa phải reproduction chuẩn. Config đã lưu: history 20, '
        '8 negatives, seed 42, tối đa 20 epoch, chọn bằng cold macro validation. SASRec dùng sampled positive/negative BCE. '
        'BERT4Rec dùng bidirectional Cloze (15% mask, thay thế 80/10/10, 10% force-last) và sampled softmax. '
        'ID chưa có tương tác train không có embedding được học; scoring ID-only dùng train-popularity fallback. '
        'Khác biệt objective với UFM phải được giữ khi diễn giải đối chứng.', '',
        '## Phần còn phải nghiệm thu', '',
        '- Sao lưu FITLAB: chỉ xem hoàn tất khi đủ 22 gói khớp hash và có restore receipt đầy đủ; theo dõi receipt ngoài repo.',
        '- Demo UFM thật: giữ bằng chứng request/checkpoint/candidate regime và nghiệm thu giao diện.',
        '- Nhiều seed và cải tiến count-aware/adaptive: chưa có kết quả train được xác minh.',
        '- CI trên GitHub, phạm vi giảng viên, Word/PDF đồng bộ và UFM All Beauty vẫn cần bằng chứng riêng.', '']
    (REPORTS/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    demo_path = REPORTS/'demo_ufm_verification.json'
    if demo_path.exists():
        demo = json.loads(demo_path.read_text(encoding='utf-8'))
        assert demo['phase']=='complete' and len(demo['cases'])==6
        assert demo['status']['model']['checkpoint_sha256']==files['ufm_full_v1.json']['checkpoint_sha256']
        with (REPORTS/'README.md').open('a',encoding='utf-8') as stream:
            stream.write('\n## Nghiệm thu phần mềm và demo\n\n'
                'Kiểm thử cuối: Windows **93 passed, 3 skipped**, Linux **96 passed, 0 skipped**, '
                'cùng 10 subtests. Ba skip trên Windows dành cho Linux flock/procfs. '
                'Log: [Windows](windows_checks_final_20261028.log), [Linux](linux_checks_final_20261028.log).\n\n'
                'Demo CPU dùng checkpoint full epoch 9 đã xác minh, catalog 767.045 sản phẩm và counts thật. '
                '**6/6 ca API đạt**: no-history, history, zero-shot, extreme-cold, cold, warm; '
                'kiểm tra trả weights/uncertainty, loại item đã chọn và đúng regime. '
                '[Bằng chứng từng request](demo_ufm_verification.json). '
                'Giao diện qua proxy FITLAB-01 đã tìm LEGO, chọn history và trả Top 5 zero-shot.\n\n'
                '![Giao diện UFM thật](demo_ufm_zero_shot.png)\n\n'
                'Sáu request đơn lẻ đo khoảng 11–53 giây trên CPU khi host có tải. '
                'Đây là số quan sát, chưa phải benchmark warm-up/p50/p95/throughput. '
                'Không sử dụng các số này như bằng chứng đạt mục tiêu latency. '
                'Ảnh catalog đầy đủ hiện dùng URL metadata; 30 ảnh demo local vẫn có chế độ offline.\n')


def plot_calibration():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = json.loads((REPORTS/'validation_v3_results.json').read_text(encoding='utf-8'))['files']
    calibration = files['ufm_full_v1_calibration.json']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, key, title in zip(axes, ['before','after'], ['Before scaling','After scaling']):
        data = calibration[key]['overall']
        bins = [b for b in data['bins'] if b['n']]
        ax.plot([0,1],[0,1],color='gray',linestyle='--',linewidth=1)
        ax.scatter([b['confidence'] for b in bins], [b['accuracy'] for b in bins],
                   s=[12+150*b['n']/data['n'] for b in bins],color='#146b87')
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean top-1 confidence',ylabel='Top-1 accuracy',
               title=f"{title} | ECE={data['ece']:.4f}")
        ax.grid(alpha=.15)
    fig.suptitle('UFM full | validation v3 audit (20,000 cases, 100 candidates/case)\nExploratory, shared-user temporal folds; confidence conditional on sampled set',fontsize=10)
    fig.savefig(REPORTS/'reliability_full_v3.png',dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
    plot_calibration()
