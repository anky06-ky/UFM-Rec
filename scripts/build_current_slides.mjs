// Editable 14-slide defense deck, built with the bundled artifact-tool runtime.
// Run with RUNTIME_NODE_MODULES set to the bundled node_modules directory.
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const root = path.resolve(import.meta.dirname, '..');
const skill = 'C:/Users/trana/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const python = 'C:/Users/trana/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const runtimeHelpers = await import(pathToFileURL(path.join(skill,'container_tools/runtime_helpers.mjs')).href);
const {Presentation,PresentationFile} = await runtimeHelpers.importRuntimeModule('@oai/artifact-tool');
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const build = path.join(root,'tmp/current_slides_20261029');
const output = path.join(root,'output/presentation/UFM_Rec_Bao_Ve_20261103.pptx');
await fs.mkdir(build,{recursive:true});
await fs.mkdir(path.dirname(output),{recursive:true});

const W=1280,H=720,FONT='Arial';
const INK='#102C3A',MUTED='#49616D',TEAL='#087D87',BLUE='#2563EB',BG='#F6F9FA',WHITE='#FFFFFF',PALE='#E7F2F2',LINE='#C9D9DE',RUST='#A35A43';
const deck=Presentation.create({slideSize:{width:W,height:H}});
deck.theme.defaultFont=FONT;

function text(slide,value,x,y,w,h,size=25,opt={}) {
  const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{style:'solid',fill:'none',width:0}});
  shape.text=value;
  shape.text.style={typeface:FONT,fontSize:size,color:opt.color||INK,bold:!!opt.bold,
    alignment:opt.align||'left',verticalAlignment:opt.valign||'middle'};
  return shape;
}
function box(slide,x,y,w,h,fill=WHITE,border=LINE){
  return slide.shapes.add({geometry:'roundRect',position:{left:x,top:y,width:w,height:h},fill,
    line:{style:'solid',fill:border,width:1},borderRadius:12});
}
function slideBase(number,title,source,notes=''){
  const slide=deck.slides.add();
  slide.background.fill=BG;
  text(slide,'UFM REC  /  ĐỒ ÁN HỌC MÁY',64,26,760,26,15,{bold:true,color:TEAL});
  text(slide,title,64,70,1138,77,38,{bold:true});
  slide.shapes.add({geometry:'rect',position:{left:64,top:154,width:1152,height:3},fill:TEAL,line:{style:'solid',fill:'none',width:0}});
  text(slide,source,64,668,1010,32,14,{color:MUTED});
  text(slide,String(number).padStart(2,'0')+' / 14',1116,668,100,32,14,{color:MUTED,align:'right'});
  slide.speakerNotes.text=notes ? `${notes}\nNguồn: ${source}` : `Nguồn: ${source}`;
  return slide;
}
function bullet(slide,value,x,y,w,h=55,size=24){
  text(slide,'•',x,y,22,h,size,{bold:true,color:TEAL});
  return text(slide,value,x+29,y,w-29,h,size);
}
function stat(slide,value,label,x,y,w){
  box(slide,x,y,w,130,WHITE,LINE);
  text(slide,value,x+18,y+18,w-36,53,36,{bold:true,color:TEAL});
  text(slide,label,x+18,y+76,w-36,40,18,{color:MUTED});
}
function nativeTable(slide,values,x,y,w,h,colWidths){
  const t=slide.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,
    values,columnWidths:colWidths});
  t.styleOptions={headerRow:true,bandedRows:true};
  t.borders.assign({style:'solid',fill:LINE,width:1});
  for(let i=0;i<values.length;i++)for(let j=0;j<values[0].length;j++){
    const cell=t.getCell(i,j);
    cell.fill=i===0?INK:(i%2===0?BG:WHITE);
    cell.text.style={typeface:FONT,fontSize:values[0].length>=4?17:20,
      color:i===0?WHITE:INK,bold:i===0};
  }
  return t;
}

// 1. Cover.
{
  const s=deck.slides.add();s.background.fill=INK;
  text(s,'UFM REC',72,70,850,55,29,{bold:true,color:'#43CAD0'});
  text(s,'Hệ khuyến nghị sản phẩm\ncold-start đa phương thức',72,185,1050,165,48,{bold:true,color:WHITE});
  text(s,'Hành vi  •  Ảnh  •  Tên và mô tả sản phẩm',72,380,1060,48,27,{color:'#D6E7EA'});
  text(s,'Nhóm: Nguyễn Thành Phong, Trần An Kỳ, Nguyễn Hoàng Hải, Hân',72,535,1070,39,22,{color:WHITE});
  text(s,'Học máy ứng dụng  |  GVHD: Phan Hồ Viết Trường  |  11/2026',72,595,1080,35,19,{color:'#BFD4D9'});
  s.speakerNotes.text='Bản bảo vệ chốt với kết quả hiện có (validation v3, seed 42). Nguồn: docs/final_report.md, docs/presentation_slides.md.';
}
// 2. Problem.
{
  const s=slideBase(2,'Sản phẩm mới và sản phẩm ít tương tác','docs/final_report.md · Chương 1');
  const data=[['0','Zero-shot','Chưa có tương tác train'],['1–5','Extreme cold','Rất ít tương tác'],['6–20','Cold','Ít tương tác'],['>20','Warm','Nhiều tương tác']];
  data.forEach((d,i)=>{const x=64+i*294;box(s,x,205,274,260,WHITE,LINE);text(s,d[0],x+22,234,230,70,48,{bold:true,color:TEAL});text(s,d[1],x+22,320,230,45,25,{bold:true});text(s,d[2],x+22,385,230,45,18,{color:MUTED});});
  bullet(s,'Ít tương tác khiến biểu diễn ID khó học; ảnh và văn bản có thể bổ sung tín hiệu.',82,510,1100,60,24);
}
// 3. Model concept.
{
  const s=slideBase(3,'UFM Rec kết hợp hai nguồn tín hiệu','docs/final_report.md · Chương 2');
  const items=[['CLIP đóng băng','Đặc trưng ảnh và văn bản'],['Lịch sử hành vi','ID và Transformer causal'],['Độ tin cậy học được','Proxy điều chỉnh trọng số'],['Đầu ra','Điểm xếp hạng sản phẩm']];
  items.forEach((d,i)=>{const y=190+i*105;box(s,90,y,1090,84,WHITE,LINE);text(s,String(i+1).padStart(2,'0'),112,y+12,62,57,34,{bold:true,color:TEAL});text(s,d[0],205,y+12,360,56,27,{bold:true});text(s,d[1],600,y+12,540,56,23,{color:MUTED});});
  s.speakerNotes.text+='\nCLIP, Transformer và BPR là thành phần kế thừa; không tuyên bố tính mới của các thành phần này.';
}
// 4. Architecture.
{
  const s=slideBase(4,'Hai nhánh của mô hình','docs/final_report.md §2.1–2.4 · src/ufm_model.py');
  box(s,65,205,555,330,WHITE,LINE);text(s,'NHÁNH SEMANTIC',95,235,490,40,25,{bold:true,color:TEAL});
  bullet(s,'CLIP text + image → adapter 128 chiều',95,295,480,75,22);
  bullet(s,'History: trung bình vector có mask',95,385,480,75,22);
  box(s,660,205,555,330,WHITE,LINE);text(s,'NHÁNH HÀNH VI',690,235,490,40,25,{bold:true,color:BLUE});
  bullet(s,'ID history → Transformer causal',690,295,480,75,22);
  bullet(s,'Unknown ID được mask trước lookup',690,385,480,75,22);
  text(s,'Ghép hai nhánh theo độ khả dụng và reliability proxy; không có history dùng fallback.',95,575,1100,60,23,{bold:true});
}
// 5. Reliability.
{
  const s=slideBase(5,'Fusion theo độ tin cậy học được','docs/final_report.md §2.3–2.4');
  bullet(s,'Mỗi nhánh tạo một proxy bất định dương; nhánh có proxy lớn nhận trọng số thấp hơn.',83,215,1090,95,25);
  bullet(s,'Khi semantic khả dụng mà ID của item zero-shot không khả dụng, trọng số chuyển sang semantic.',83,328,1090,100,25);
  bullet(s,'Khi cả hai nhánh không khả dụng, hệ thống dùng popularity của tập train.',83,450,1090,90,25);
  box(s,86,575,1080,55,PALE,PALE);text(s,'Proxy này chưa được chứng minh là posterior Bayes hoặc xác suất mua hàng.',105,580,1035,40,21,{bold:true,color:INK});
}
// 6. Protocol.
{
  const s=slideBase(6,'Dữ liệu và giao thức đánh giá','reports/continuation_20261028/summary.json');
  stat(s,'767.045','sản phẩm Toys & Games',65,210,260);
  stat(s,'11,57 triệu','tương tác đã làm sạch',355,210,260);
  stat(s,'40.000','ca validation v3',645,210,260);
  stat(s,'100','ứng viên mỗi ca',935,210,260);
  bullet(s,'Chia thời gian 80/10/10; mỗi regime có 10.000 ca, một positive và 99 negatives.',80,390,1090,80,23);
  bullet(s,'Checkpoint chọn bằng validation v1; v3 là đánh giá lại trên quần thể liên quan.',80,485,1090,80,23);
  text(s,'Sampling theo regime target và thống kê cuối split; không tương đương xếp hạng toàn catalog.',110,585,1050,43,20,{bold:true,color:RUST});
}
// 7. Ranking results: native chart.
{
  const s=slideBase(7,'Cold macro NDCG@10 trên validation v3','reports/nontraining_20261029/evidence.json · 40.000 ca · seed 42');
  applyPresentationChartFont(s.charts.add('bar',{position:{left:55,top:190,width:815,height:430},
    categories:['Popularity','Collaborative-only','Image-only','Fixed fusion','Text-only','UFM full','No uncertainty','No cross-align','Semantic-only','TF-IDF'],
    series:[{name:'Cold macro NDCG@10',values:[0.042527,0.070003,0.149632,0.181305,0.181982,0.182056,0.182253,0.183393,0.190036,0.195682],fill:TEAL}],
    barOptions:{direction:'bar',grouping:'clustered',gapWidth:45},hasLegend:false,
    xAxis:{numberFormatCode:'0.000'},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000'}}),{fontFamily:FONT});
  box(s,900,205,290,310,WHITE,LINE);text(s,'KẾT QUẢ CHÍNH',920,227,245,45,19,{bold:true,color:TEAL});
  text(s,'TF-IDF\n0,195682',920,285,245,65,25,{bold:true});
  text(s,'Semantic-only\n0,190036',920,370,245,65,25,{bold:true});
  text(s,'UFM full\n0,182056',920,455,245,65,25,{bold:true,color:RUST});
  text(s,'Full chưa vượt hai đối chứng nội dung trên chỉ số chính.',910,555,280,70,20,{color:RUST});
  s.speakerNotes.text+='\nTF-IDF dùng 10 history item; UFM dùng 20. Bảng là so sánh hệ thống, không gán chênh lệch cho riêng thành phần.';
}
// 8. Train-count pattern: native table and chart.
{
  const s=slideBase(8,'Chênh lệch thay đổi theo train count','reports/continuation_20261028/count_effect_v3.json · seed 42');
  nativeTable(s,[['Train count','Full − semantic-only'],['0','−0,002424'],['1–5','−0,024101'],['6–20','+0,002586'],['21+','+0,014865']],68,205,510,315,[195,315]);
  applyPresentationChartFont(s.charts.add('bar',{position:{left:615,top:195,width:570,height:320},
    categories:['0','1–5','6–20','21+'],series:[{name:'Δ NDCG@10',values:[-0.002424,-0.024101,0.002586,0.014865],fill:TEAL}],
    barOptions:{direction:'column',grouping:'clustered',gapWidth:70},hasLegend:false,
    yAxis:{numberFormatCode:'0.000'},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000'}}),{fontFamily:FONT});
  text(s,'Mỗi nhóm 10.000 ca. Nhóm 1–5 giảm nhiều nhất; chênh lệch chưa chứng minh ID/gate là nguyên nhân.',80,552,1090,82,22,{bold:true});
}
// 9. Calibration.
{
  const s=slideBase(9,'Calibration thăm dò trên validation v3','reports/continuation_20261028/summary.json · T=2,012405');
  nativeTable(s,[['Chỉ số audit','Trước','Sau'],['ECE','0,082061','0,037478'],['NLL','4,497667','4,258086'],['Multiclass Brier','0,969342','0,960605']],70,235,735,315,[325,205,205]);
  box(s,840,235,355,315,WHITE,LINE);
  bullet(s,'Fit trên 20.000 ca sớm; audit trên 20.000 ca muộn.',860,258,320,78,20);
  bullet(s,'Temperature không đổi thứ hạng.',860,355,320,67,20);
  bullet(s,'Confidence có điều kiện trên 100 ứng viên.',860,445,320,70,20);
  s.speakerNotes.text+='\nNguồn phương pháp temperature scaling: Guo et al. (2017), https://proceedings.mlr.press/v70/guo17a.html';
}
// 10. Runtime.
{
  const s=slideBase(10,'Độ trễ UFM trên Windows CPU','reports/continuation_20261028/windows_ufm_benchmark.json');
  stat(s,'21,25 s','p50 (10 request)',80,240,340);
  stat(s,'26,35 s','p95 (10 request)',470,240,340);
  stat(s,'2.551 MiB','RSS cao nhất quan sát',860,240,340);
  bullet(s,'Hai request đồng thời trả một 200 và một 429; server xử lý một suy luận tại một thời điểm.',80,435,1100,100,23);
  text(s,'10 request chưa đủ cho p99 ổn định; peak VRAM train 16 GiB chưa được đo.',110,572,1040,55,22,{bold:true,color:RUST});
}
// 11. Demo and UI proof.
{
  const s=slideBase(11,'Demo UFM thật và nghiệm thu giao diện','reports/nontraining_20261029/demo_ui_end_to_end_20261029.json');
  box(s,65,210,580,365,WHITE,LINE);
  bullet(s,'Catalog thật: 767.045 sản phẩm',88,240,530,62,23);
  bullet(s,'6/6 ca API và 4/4 ca UI đạt',88,319,530,62,23);
  bullet(s,'JSON tải về khớp request, model hash và 5 kết quả',88,398,530,80,22);
  bullet(s,'Ảnh lỗi vẫn hiện placeholder; mất mạng hiện lỗi',88,490,530,70,22);
  s.images.add({blob:await fs.readFile(path.join(root,'reports/nontraining_20261029/demo_ui_zero_shot_20261029.png')),
    contentType:'image/png',
    alt:'Ảnh chụp demo UFM zero-shot trên Windows với ảnh bị chặn và placeholder',fit:'cover',
    position:{left:685,top:205,width:490,height:380}});
  text(s,'Ảnh UI là bằng chứng thao tác; điểm xếp hạng không phải xác suất mua.',88,602,1075,48,20,{color:MUTED});
}
// 12. Historical v2 table.
{
  const s=slideBase(12,'Kết quả evaluation v2 giữ riêng','reports/ufm_evaluation_v2_20261027/README.md');
  nativeTable(s,[['Nhóm','Số ca','Validation','Test'],['Zero-shot','10.000','0,156641','0,157281'],
    ['Extreme cold','10.000','0,191567','0,195261'],['Cold','10.000','0,197946','0,198373'],
    ['Warm','10.000','0,110375','0,113859'],['Cold macro','30.000','0,182051','0,183638'],
    ['Overall','40.000','0,164132','0,166193']],72,195,1110,370,[345,180,292,293]);
  text(s,'Test v2 đã được xem. Candidates v2 khác v3; không ghép các điểm này vào bảng v3.',85,590,1090,65,20,{bold:true,color:RUST});
}
// 13. Conclusion.
{
  const s=slideBase(13,'Kết luận từ bằng chứng hiện có','docs/final_report.md · Chương 6');
  bullet(s,'Đã có hệ thống ba nguồn tín hiệu, checkpoint, đánh giá cùng candidates và demo UFM.',78,210,1110,85,25);
  bullet(s,'Full UFM đạt 0,182056 Cold macro v3; chưa vượt TF-IDF 0,195682 và semantic-only 0,190036.',78,322,1110,100,25);
  bullet(s,'Nhóm 1–5 tương tác là ưu tiên phân tích; hai cải tiến count-aware/adaptive dropout chưa có kết quả train.',78,455,1110,110,25);
  box(s,90,585,1090,59,PALE,PALE);text(s,'Hướng tiếp theo: 15 lượt đa seed đã khóa, holdout độc lập.',110,591,1050,45,22,{bold:true});
}
// 14. Thanks.
{
  const s=deck.slides.add();s.background.fill=INK;
  text(s,'Cảm ơn Thầy Cô đã lắng nghe',85,235,1110,95,48,{bold:true,color:WHITE});
  text(s,'Nhóm sẵn sàng trao đổi về kết quả và giới hạn thực nghiệm.',85,365,1110,65,29,{color:'#D6E7EA'});
  text(s,'UFM Rec · 11/2026',85,595,1030,35,20,{color:'#43CAD0'});
  s.speakerNotes.text='Mời câu hỏi. Không tuyên bố hai cải tiến đã thắng hoặc có test độc lập v3.';
}

if(deck.slides.items.length!==14)throw new Error('Expected 14 slides');
const preview=path.join(build,'rendered');await fs.mkdir(preview,{recursive:true});
for(const [i,slide] of deck.slides.items.entries()){
  const png=await deck.export({slide,format:'png',scale:1});
  await fs.writeFile(path.join(preview,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
const result=await finalizePresentation({
  workspaceDir:root,candidatePath:candidate,finalPath:output,pythonExecutable:python,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry',
    '--validate-heading-fit','--require-native-table-slide','8','--require-native-table-slide','9',
    '--require-native-table-slide','12'],
  explicitTotalSlideCount:14,requiredNativeTableOwnerSlides:[8,9,12],
  requiredNativeChartOwnerSlides:[7,8],materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
  receiptPath:path.join(build,`validation-${Date.now()}.json`),
});
console.log(JSON.stringify({slides:14,output,validation:result},null,2));
