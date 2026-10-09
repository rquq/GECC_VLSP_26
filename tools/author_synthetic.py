#!/usr/bin/env python3
"""Author a private, deterministic Vietnamese GEC contrast pool.

Run ``python tools/author_synthetic.py`` from any working directory. Only the
private synthetic_pool.json is replaced; organizer text and gold are never read.
Each marked, explicitly authored sentence yields one local corruption and one
exact clean counterpart, not a lexical Cartesian product. The six class-1 rules
cover missing main action predicates, embedded subject/predicate inversion,
three subordinate-clause relations, and a split coordinated subject. Class 2 covers eight
noun/adjective phrase order and attachment rules, rather than numeral order alone.
Class 3 covers eight overt function-word incompatibilities/repetitions. Class 4
covers eight mechanical punctuation rules, not question/declarative conversion.

This is assistant review, NOT human adjudication or evidence of model benefit.
The class boundary between clause structure and function words remains a review
risk; clause examples therefore remove/move constituents, not prepositions.
Subject ellipsis, topic-fronting, acceptable idioms, optional conjunctions, stylistic
rewrites, semantic substitutions, and preposition-deletion examples are excluded.
Punctuation examples assume ordinary edited prose (not poetry/chat typography).
The extra legal fronted/prepositional/subordinate sentences are clean-only
families. All families are training-only; consumers must preserve original-only
validation and select whole provenance-aware examples without claiming this pool
is human-adjudicated. Stable family IDs are append-only rule/index identifiers.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "dataset/newdataset/synthetic_pool.json"
REVIEW_STATUS = "assistant_reviewed_not_human_adjudicated"

# A single [marked span] is replaced, never a global lexical substitution.
# Explicit bad spans make the intended contrast inspectable before generation.
# Entries must remain in order: their ordinal is part of the stable family ID.
GROUPS = [
    (1, "missing_main_action_predicate", "Hai danh ngữ còn lại không tạo thành câu trần thuật hoàn chỉnh; khôi phục vị ngữ hành động đã bị bỏ trong đối chứng.", [
        ("Bác sĩ [đã kiểm tra] hồ sơ trước ca trực.", ""),
        ("Cô Hạnh [đang hướng dẫn] học sinh làm thí nghiệm.", ""),
        ("Người thủ thư [đã đặt] sách lên kệ mới.", ""),
        ("Anh Nam [sẽ sửa] chiếc xe vào chiều mai.", ""),
        ("Đội cứu hộ [đã đưa] người dân đến nơi an toàn.", ""),
        ("Nhóm khảo sát [đang đo] độ sâu của hồ.", ""),
        ("Bà Mai [đã tưới] vườn rau trước bữa sáng.", ""),
        ("Tổ kỹ thuật [sẽ kiểm tra] đường dây vào thứ hai.", ""),
        ("Các đầu bếp [đang chuẩn bị] bữa trưa cho đoàn.", ""),
        ("Chú bảo vệ [đã khóa] cổng sau giờ tan học.", ""),
        ("Đội bóng [sẽ sử dụng] sân phía đông để tập luyện.", ""),
        ("Chị Hoa [đang phân loại] hạt giống trong kho.", ""),
        ("Người hướng dẫn [đã giới thiệu] lịch trình của chuyến đi.", ""),
        ("Ban tổ chức [sẽ công bố] kết quả vào sáng mai.", ""),
    ]),
    (1, "embedded_subject_after_aspect", "Chủ ngữ của mệnh đề bổ ngữ bị đặt sau từ thời-thể và trước động từ; chuyển chủ ngữ lên trước vị ngữ.", [
        ("Lan biết rằng [Minh đã] nộp bản báo cáo trước buổi họp.", "đã Minh"),
        ("Cô giáo thấy rằng [Nam đang] tiến bộ trong môn toán.", "đang Nam"),
        ("Người quản lý xác nhận rằng [tài xế đã] kiểm tra phanh trước khi khởi hành.", "đã tài xế"),
        ("Bà Mai cho biết rằng [nông dân đã] thu hoạch ngô vào sáng nay.", "đã nông dân"),
        ("Anh Tuấn thông báo rằng [chị Hà sẽ] mở quầy hàng vào lúc 7 giờ.", "sẽ chị Hà"),
        ("Minh nhận thấy rằng [đội vệ sinh đang] quét sân sau buổi lễ.", "đang đội vệ sinh"),
        ("Mẹ biết rằng [em bé đã] xếp đồ chơi vào chiếc hộp màu xanh.", "đã em bé"),
        ("Khách hàng xác nhận rằng [nhân viên đã] gửi hóa đơn vào sáng nay.", "đã nhân viên"),
        ("Hạnh cho biết rằng [cô giáo đang] đọc truyện cho cả lớp nghe.", "đang cô giáo"),
        ("Chị Hoa thông báo rằng [ông Bình sẽ] trồng cây ở góc vườn phía nam.", "sẽ ông Bình"),
        ("Chủ nhà xác nhận rằng [người thợ đã] lắp cửa cho phòng làm việc.", "đã người thợ"),
        ("Lan thấy rằng [tình nguyện viên đang] phát nước tại cổng hội trường.", "đang tình nguyện viên"),
        ("Giám đốc cho biết rằng [nhóm thiết kế đã] hoàn thành bản vẽ vào thứ sáu.", "đã nhóm thiết kế"),
        ("Cô Hà thông báo rằng [đoàn kiểm tra sẽ] đến trường vào tuần sau.", "sẽ đoàn kiểm tra"),
    ]),
    (1, "temporal_clause_predicate_intrusion", "Vị ngữ mệnh đề thời gian bị chuyển sau dấu phẩy và chủ ngữ mệnh đề chính bị chuyển lên trước; khôi phục hai cặp chủ-vị.", [
        ("Khi [chuông reo, học sinh] ra sân.", "chuông học sinh, reo"),
        ("Khi [mưa tạnh, người thợ] tiếp tục sửa mái nhà.", "mưa người thợ, tạnh"),
        ("Khi [tàu cập bến, hành khách] xuống cầu tàu.", "tàu hành khách, cập bến"),
        ("Khi [đèn bật sáng, diễn viên] bước lên sân khấu.", "đèn diễn viên, bật sáng"),
        ("Khi [nước sôi, mẹ] cho rau vào nồi.", "nước mẹ, sôi"),
        ("Khi [cửa mở, khách hàng] bước vào cửa tiệm.", "cửa khách hàng, mở"),
        ("Khi [buổi họp kết thúc, thư ký] gửi biên bản cho mọi người.", "buổi họp thư ký, kết thúc"),
        ("Khi [mặt trời lặn, ngư dân] trở về bến.", "mặt trời ngư dân, lặn"),
        ("Khi [đồng hồ điểm 8 giờ, bác sĩ] bắt đầu buổi khám.", "đồng hồ bác sĩ, điểm 8 giờ"),
        ("Khi [gió ngừng thổi, đội khảo sát] dựng lều.", "gió đội khảo sát, ngừng thổi"),
        ("Khi [tiết học bắt đầu, cô giáo] kiểm tra sĩ số.", "tiết học cô giáo, bắt đầu"),
        ("Khi [máy in dừng lại, nhân viên] lấy tập tài liệu ra.", "máy in nhân viên, dừng lại"),
        ("Khi [cây cầu thông xe, đoàn xe] rời trạm chờ.", "cây cầu đoàn xe, thông xe"),
        ("Khi [trận đấu kết thúc, trọng tài] ghi kết quả vào biên bản.", "trận đấu trọng tài, kết thúc"),
    ]),
    (1, "conditional_clause_predicate_intrusion", "Vị ngữ điều kiện bị chuyển vào mệnh đề kết quả và chủ ngữ kết quả bị chuyển lên trước dấu phẩy; khôi phục hai cặp chủ-vị.", [
        ("Nếu [trời lạnh, đoàn khách] sẽ mặc áo khoác.", "trời đoàn khách, lạnh"),
        ("Nếu [máy hỏng, kỹ thuật viên] sẽ thay linh kiện.", "máy kỹ thuật viên, hỏng"),
        ("Nếu [đường ngập, tài xế] sẽ chọn lối khác.", "đường tài xế, ngập"),
        ("Nếu [giá giảm, chị Mai] sẽ mua thêm vải.", "giá chị Mai, giảm"),
        ("Nếu [mực hết, nhân viên] sẽ thay hộp mực.", "mực nhân viên, hết"),
        ("Nếu [vé còn, gia đình tôi] sẽ đi xem kịch.", "vé gia đình tôi, còn"),
        ("Nếu [pin yếu, anh Tuấn] sẽ sạc điện thoại.", "pin anh Tuấn, yếu"),
        ("Nếu [nước cạn, người làm vườn] sẽ bật máy bơm.", "nước người làm vườn, cạn"),
        ("Nếu [gió mạnh, thủy thủ] sẽ hạ cánh buồm.", "gió thủy thủ, mạnh"),
        ("Nếu [sân khô, các cầu thủ] sẽ tập ngoài trời.", "sân các cầu thủ, khô"),
        ("Nếu [đèn tắt, người quản lý] sẽ kiểm tra cầu dao.", "đèn người quản lý, tắt"),
        ("Nếu [bột đủ, cô Lan] sẽ làm thêm bánh.", "bột cô Lan, đủ"),
        ("Nếu [xe đến muộn, hành khách] sẽ chờ ở sảnh.", "xe hành khách, đến muộn"),
        ("Nếu [hồ sơ đầy đủ, cán bộ] sẽ tiếp nhận đơn.", "hồ sơ cán bộ, đầy đủ"),
    ]),
    (1, "concessive_clause_predicate_intrusion", "Vị ngữ nhượng bộ bị chuyển sang mệnh đề chính và chủ ngữ mệnh đề chính bị chuyển sang mệnh đề phụ; giữ quan hệ mặc dù-vẫn.", [
        ("Mặc dù [trời mưa, đội cứu hộ] vẫn lên đường.", "trời đội cứu hộ, mưa"),
        ("Mặc dù [đường xa, đoàn tình nguyện] vẫn đến đúng giờ.", "đường đoàn tình nguyện, xa"),
        ("Mặc dù [gió mạnh, người thợ] vẫn sửa cửa sổ.", "gió người thợ, mạnh"),
        ("Mặc dù [sân nhỏ, các em] vẫn tập thể dục đều đặn.", "sân các em, nhỏ"),
        ("Mặc dù [giá cao, chị Hoa] vẫn mua loại thuốc này.", "giá chị Hoa, cao"),
        ("Mặc dù [việc nhiều, anh Nam] vẫn đọc sách mỗi tối.", "việc anh Nam, nhiều"),
        ("Mặc dù [đêm lạnh, nhóm quan sát] vẫn theo dõi các vì sao.", "đêm nhóm quan sát, lạnh"),
        ("Mặc dù [phòng chật, người thủ thư] vẫn sắp xếp đủ chỗ ngồi.", "phòng người thủ thư, chật"),
        ("Mặc dù [sóng lớn, thuyền trưởng] vẫn giữ đúng hướng đi.", "sóng thuyền trưởng, lớn"),
        ("Mặc dù [máy cũ, kỹ thuật viên] vẫn bảo dưỡng hằng tháng.", "máy kỹ thuật viên, cũ"),
        ("Mặc dù [thời gian ít, cô giáo] vẫn giải thích từng bước.", "thời gian cô giáo, ít"),
        ("Mặc dù [trận đấu khó, đội bóng] vẫn giữ tinh thần bình tĩnh.", "trận đấu đội bóng, khó"),
        ("Mặc dù [kinh phí hạn chế, ban tổ chức] vẫn chuẩn bị đủ nước uống.", "kinh phí ban tổ chức, hạn chế"),
        ("Mặc dù [lối đi hẹp, nhân viên] vẫn chuyển được chiếc bàn.", "lối đi nhân viên, hẹp"),
    ]),
    (1, "predicate_splits_coordinated_subject", "Từ chỉ thời-thể của vị ngữ chen vào hai thành phần chủ ngữ phối hợp; trả về sau toàn bộ chủ ngữ.", [
        ("[Lan và Minh đã] hoàn thành mô hình cây cầu.", "Lan đã và Minh"),
        ("[Hoa và Hạnh đang] sắp xếp tài liệu trong phòng họp.", "Hoa đang và Hạnh"),
        ("[Nam và Tuấn sẽ] kiểm tra thiết bị vào sáng mai.", "Nam sẽ và Tuấn"),
        ("[Bác sĩ và y tá đã] chuẩn bị phòng khám.", "Bác sĩ đã và y tá"),
        ("[Thầy Sơn và cô Mai đang] hướng dẫn đội tuyển.", "Thầy Sơn đang và cô Mai"),
        ("[Ông Bình và bà Hà sẽ] trồng thêm hai cây cam.", "Ông Bình sẽ và bà Hà"),
        ("[Đội trưởng và huấn luyện viên đã] thống nhất lịch tập.", "Đội trưởng đã và huấn luyện viên"),
        ("[Người lái xe và phụ xe đang] kiểm tra hành lý.", "Người lái xe đang và phụ xe"),
        ("[Thư ký và kế toán sẽ] đối chiếu các hóa đơn.", "Thư ký sẽ và kế toán"),
        ("[Anh Dũng và chị Liên đã] dựng xong gian hàng.", "Anh Dũng đã và chị Liên"),
        ("[Tổ cơ khí và tổ điện đang] bảo dưỡng dây chuyền.", "Tổ cơ khí đang và tổ điện"),
        ("[Mai và Phương sẽ] trình bày kết quả khảo sát.", "Mai sẽ và Phương"),
        ("[Đầu bếp và phụ bếp đã] chuẩn bị đủ nguyên liệu.", "Đầu bếp đã và phụ bếp"),
        ("[Người quản lý và bảo vệ đang] kiểm tra lối thoát hiểm.", "Người quản lý đang và bảo vệ"),
    ]),
    (2, "numeral_classifier_order", "Số từ đứng trước loại từ trong danh ngữ đếm; đảo loại từ ra trước số từ làm sai cấu trúc.", [
        ("Lan mượn [ba quyển] sách ở thư viện.", "quyển ba"),
        ("Nam mang [hai chiếc] ghế vào phòng họp.", "chiếc hai"),
        ("Mẹ mua [bốn quả] cam cho bữa tráng miệng.", "quả bốn"),
        ("Tổ thợ lắp [năm bộ] đèn trên hành lang.", "bộ năm"),
        ("Cô giáo phát [sáu tờ] giấy cho nhóm của Minh.", "tờ sáu"),
        ("Ông Bình trồng [bảy cây] xoài ở vườn sau.", "cây bảy"),
        ("Nhân viên xếp [tám thùng] hàng vào kho.", "thùng tám"),
        ("Chị Hà chuẩn bị [chín chiếc] khăn cho đội văn nghệ.", "chiếc chín"),
        ("Đầu bếp luộc [mười quả] trứng cho bữa sáng.", "quả mười"),
        ("Anh Tuấn mua [hai đôi] giày cho các con.", "đôi hai"),
    ]),
    (2, "demonstrative_before_noun_head", "Từ chỉ định hậu danh ngữ bị đặt trước loại từ và danh từ; trả về sau danh từ.", [
        ("Tôi sẽ giặt [chiếc áo này] vào chiều nay.", "này chiếc áo"),
        ("Lan đã đọc [cuốn truyện ấy] trong kỳ nghỉ.", "ấy cuốn truyện"),
        ("Người thợ sẽ sửa [cánh cửa đó] trước thứ sáu.", "đó cánh cửa"),
        ("Bà Mai muốn giữ [bức ảnh này] trong album.", "này bức ảnh"),
        ("Anh Nam đang kiểm tra [chiếc máy đó] ở xưởng.", "đó chiếc máy"),
        ("Cô giáo đặt [tấm bản đồ ấy] lên bàn.", "ấy tấm bản đồ"),
        ("Minh đã tưới [chậu hoa này] từ sáng sớm.", "này chậu hoa"),
        ("Chị Hà sẽ chuyển [cái bàn đó] sang phòng bên.", "đó cái bàn"),
        ("Tôi đã lưu [tệp tài liệu này] trên máy tính.", "này tệp tài liệu"),
        ("Người thủ thư đánh dấu [trang sách ấy] bằng giấy màu.", "ấy trang sách"),
    ]),
    (2, "adjective_splits_classifier_head", "Tính từ miêu tả bị chen giữa loại từ và danh từ; tính từ phải theo sau danh từ ở các danh ngữ này.", [
        ("Đoàn khách nghỉ trong [căn phòng rộng] ở tầng hai.", "căn rộng phòng"),
        ("Em bé cầm [quả bóng đỏ] trên sân.", "quả đỏ bóng"),
        ("Lan mặc [chiếc áo xanh] đến lớp.", "chiếc xanh áo"),
        ("Chúng tôi đi qua [cây cầu dài] gần bến cảng.", "cây dài cầu"),
        ("Minh đặt sách lên [cái bàn thấp] cạnh cửa sổ.", "cái thấp bàn"),
        ("Người thợ mang [chiếc hộp nặng] vào kho.", "chiếc nặng hộp"),
        ("Bà Mai hái [quả cam chín] trong vườn.", "quả chín cam"),
        ("Anh Tuấn dựng [chiếc xe mới] trước cổng.", "chiếc mới xe"),
        ("Cô giáo chọn [bức tranh đẹp] để treo trong lớp.", "bức đẹp tranh"),
        ("Chị Hoa mua [chiếc khăn mềm] cho mẹ.", "chiếc mềm khăn"),
    ]),
    (2, "degree_modifier_after_adjective", "Phó từ mức độ rất bị đặt sau tính từ mà không có bổ ngữ; đưa về trước tính từ.", [
        ("Phòng đọc hôm nay [rất yên tĩnh].", "yên tĩnh rất"),
        ("Con đường mới [rất rộng].", "rộng rất"),
        ("Bát canh mẹ nấu [rất nóng].", "nóng rất"),
        ("Bản hướng dẫn này [rất rõ ràng].", "rõ ràng rất"),
        ("Chiếc túi của Lan [rất nhẹ].", "nhẹ rất"),
        ("Nước trong hồ [rất sạch].", "sạch rất"),
        ("Cây xoài ở góc vườn [rất cao].", "cao rất"),
        ("Lối vào thư viện [rất thuận tiện].", "thuận tiện rất"),
        ("Người hướng dẫn của đoàn [rất nhiệt tình].", "nhiệt tình rất"),
        ("Mẫu vải chị Hà chọn [rất mềm].", "mềm rất"),
    ]),
    (2, "possessive_complement_before_head", "Bổ ngữ sở hữu có của bị chen trước danh từ trung tâm, làm đứt tổ hợp loại từ-danh từ.", [
        ("Tôi đã trả [chiếc xe của Nam] vào chiều qua.", "chiếc của Nam xe"),
        ("Cô giáo giữ [cuốn vở của Lan] trên bàn.", "cuốn của Lan vở"),
        ("Minh đã tìm thấy [chiếc mũ của Tuấn] trong lớp.", "chiếc của Tuấn mũ"),
        ("Người thợ sửa [cái tủ của bà Mai] vào sáng nay.", "cái của bà Mai tủ"),
        ("Chị Hoa giặt [chiếc áo của Hạnh] bằng tay.", "chiếc của Hạnh áo"),
        ("Anh Dũng mang [chiếc máy ảnh của Phương] đến hội trường.", "chiếc của Phương máy ảnh"),
        ("Tôi đã đọc [bức thư của ông Bình] tối qua.", "bức của ông Bình thư"),
        ("Nhân viên chuyển [cái bàn của cô Hà] sang phòng mới.", "cái của cô Hà bàn"),
        ("Người thủ thư cất [tấm thẻ của Minh] vào ngăn kéo.", "tấm của Minh thẻ"),
        ("Mẹ đã rửa [chiếc cốc của bé An] sau bữa sáng.", "chiếc của bé An cốc"),
    ]),
    (2, "plural_marker_after_classifier", "Dấu hiệu số nhiều những bị đặt sau loại từ; đưa những ra trước loại từ.", [
        ("Lan đã giặt [những chiếc] khăn trong giỏ.", "chiếc những"),
        ("Nam đang đọc [những cuốn] sách trên kệ bên trái.", "cuốn những"),
        ("Người làm vườn đã tưới [những chậu] hoa ở ban công.", "chậu những"),
        ("Cô giáo phát [những tờ] giấy màu cho cả lớp.", "tờ những"),
        ("Anh Tuấn kiểm tra [những chiếc] xe trong sân.", "chiếc những"),
        ("Nhân viên chuyển [những thùng] hàng đến kho mới.", "thùng những"),
        ("Bà Mai hái [những quả] cam đã chín.", "quả những"),
        ("Đoàn khách tham quan [những ngôi] nhà cổ ở làng.", "ngôi những"),
        ("Minh đã sắp xếp [những bộ] dụng cụ vào hộp.", "bộ những"),
        ("Chị Hà phơi [những chiếc] áo ở sân sau.", "chiếc những"),
    ]),
    (2, "comparative_marker_before_adjective", "Trong phép so sánh trực tiếp tính từ-hơn-đối tượng, hơn bị chuyển ra trước tính từ.", [
        ("Cây bàng [cao hơn] cây ổi ở cạnh cổng.", "hơn cao"),
        ("Chiếc hộp xanh [nặng hơn] chiếc hộp trắng.", "hơn nặng"),
        ("Phòng họp mới [rộng hơn] phòng họp cũ.", "hơn rộng"),
        ("Con đường phía đông [ngắn hơn] con đường phía tây.", "hơn ngắn"),
        ("Nước trong bình [ấm hơn] nước trong cốc.", "hơn ấm"),
        ("Tấm kính này [dày hơn] tấm kính kia.", "hơn dày"),
        ("Chiếc khăn lụa [mềm hơn] chiếc khăn vải.", "hơn mềm"),
        ("Sân trường [sạch hơn] sân kho sau cơn mưa.", "hơn sạch"),
        ("Phòng đọc [yên tĩnh hơn] sảnh chờ.", "hơn yên tĩnh"),
        ("Bản in mới [rõ hơn] bản in cũ.", "hơn rõ"),
    ]),
    (2, "topic_complement_before_noun_head", "Bổ ngữ đề tài có về bị chen giữa loại từ và danh từ trung tâm; trả về sau danh từ.", [
        ("Lan mượn [cuốn sách về lịch sử] ở thư viện.", "cuốn về lịch sử sách"),
        ("Nam đã đọc [bài báo về giao thông] sáng nay.", "bài về giao thông báo"),
        ("Cô giáo chuẩn bị [bài giảng về thực vật] cho lớp 6.", "bài về thực vật giảng"),
        ("Minh đang viết [bản báo cáo về nguồn nước] của xã.", "bản về nguồn nước báo cáo"),
        ("Đoàn khách xem [bộ phim về biển đảo] tại hội trường.", "bộ về biển đảo phim"),
        ("Chị Hà giữ [cuốn sổ về chi tiêu] trong ngăn bàn.", "cuốn về chi tiêu sổ"),
        ("Anh Tuấn trình bày [bản kế hoạch về bảo dưỡng] vào thứ hai.", "bản về bảo dưỡng kế hoạch"),
        ("Học sinh nghe [câu chuyện về lòng dũng cảm] trong giờ sinh hoạt.", "câu về lòng dũng cảm chuyện"),
        ("Bác sĩ phát [tờ hướng dẫn về dinh dưỡng] cho người bệnh.", "tờ về dinh dưỡng hướng dẫn"),
        ("Người quản lý lưu [tệp dữ liệu về doanh thu] trên máy chủ.", "tệp về doanh thu dữ liệu"),
    ]),
    (3, "possessive_marker_replaced_by_complementizer", "Rằng không nối danh từ với người sở hữu; dùng của cho quan hệ sở hữu đã nêu.", [
        ("Chiếc xe [của] Nam đang ở trong sân.", "rằng"),
        ("Cuốn vở [của] Lan nằm trên bàn học.", "rằng"),
        ("Chiếc mũ [của] Minh đã được giặt sạch.", "rằng"),
        ("Căn nhà [của] bà Mai ở gần bưu điện.", "rằng"),
        ("Máy ảnh [của] Phương được cất trong tủ.", "rằng"),
        ("Chiếc túi [của] Hạnh có hai ngăn nhỏ.", "rằng"),
        ("Đàn ghi ta [của] Tuấn đã được lên dây.", "rằng"),
        ("Tập bản vẽ [của] cô Hà nằm trong chiếc hộp xanh.", "rằng"),
        ("Bộ dụng cụ [của] anh Dũng đã được kiểm tra.", "rằng"),
        ("Chiếc cốc [của] bé An có hình một con mèo.", "rằng"),
    ]),
    (3, "coordination_marker_replaced_by_complementizer", "Rằng không phối hợp hai danh ngữ đồng chức năng; dùng và, giữ nguyên hai đối tượng.", [
        ("Mẹ mua gạo [và] muối ở cửa hàng gần nhà.", "rằng"),
        ("Lan mang bút [và] vở đến lớp.", "rằng"),
        ("Nam đã rửa cốc [và] đĩa sau bữa tối.", "rằng"),
        ("Cô giáo giới thiệu sách [và] bản đồ cho cả lớp.", "rằng"),
        ("Nhân viên kiểm tra vé [và] giấy tờ tại cổng.", "rằng"),
        ("Người thợ chuẩn bị búa [và] đinh trước khi sửa cửa.", "rằng"),
        ("Đầu bếp cắt cà rốt [và] khoai tây cho món súp.", "rằng"),
        ("Minh cất kéo [và] thước vào ngăn bàn.", "rằng"),
        ("Chị Hà đặt hoa [và] nến lên bàn lễ.", "rằng"),
        ("Đội cứu hộ mang nước [và] lương thực đến trạm.", "rằng"),
    ]),
    (3, "instrumental_preposition_with_possessive_marker", "Của bị chèn ngay sau bằng trước danh từ công cụ không có vật sở hữu; bỏ hư từ thừa.", [
        ("Lan viết nháp [bằng] bút chì.", "bằng của"),
        ("Nam buộc thùng hàng [bằng] dây thừng.", "bằng của"),
        ("Người thợ cắt tấm ván [bằng] cưa tay.", "bằng của"),
        ("Chị Hà gói quà [bằng] giấy màu.", "bằng của"),
        ("Minh đo chiều dài chiếc bàn [bằng] thước dây.", "bằng của"),
        ("Đầu bếp khuấy bột [bằng] thìa gỗ.", "bằng của"),
        ("Nhân viên đóng dấu hồ sơ [bằng] con dấu mới.", "bằng của"),
        ("Cô giáo minh họa bài học [bằng] hình vẽ.", "bằng của"),
        ("Anh Tuấn lau cửa kính [bằng] khăn mềm.", "bằng của"),
        ("Bà Mai che giỏ rau [bằng] tấm vải sạch.", "bằng của"),
    ]),
    (3, "topic_preposition_with_complementizer", "Rằng bị chèn sau về trước danh ngữ đề tài, không có mệnh đề được dẫn; bỏ rằng.", [
        ("Cả lớp thảo luận [về] an toàn giao thông.", "về rằng"),
        ("Lan đọc tài liệu [về] cách trồng rau.", "về rằng"),
        ("Nam trình bày báo cáo [về] chất lượng nước.", "về rằng"),
        ("Cô giáo kể chuyện [về] tuổi thơ của nhà văn.", "về rằng"),
        ("Nhóm nghiên cứu trao đổi [về] kết quả đo đạc.", "về rằng"),
        ("Đoàn khách xem phim [về] đời sống trên đảo.", "về rằng"),
        ("Chị Hà tìm thông tin [về] lịch tàu chạy.", "về rằng"),
        ("Bác sĩ tư vấn [về] chế độ dinh dưỡng.", "về rằng"),
        ("Anh Tuấn viết hướng dẫn [về] bảo dưỡng xe.", "về rằng"),
        ("Người quản lý thông báo [về] thời gian mở cửa.", "về rằng"),
    ]),
    (3, "modal_with_complementizer_before_action", "Rằng không thể chen giữa phải và động từ hành động trong vị ngữ nghĩa vụ; giữ phải và bỏ rằng.", [
        ("Học sinh [phải] đội mũ bảo hiểm khi đi xe đạp điện.", "phải rằng"),
        ("Nhân viên [phải] khóa cửa trước khi rời kho.", "phải rằng"),
        ("Lan [phải] nộp đơn trước ngày 12 tháng 5.", "phải rằng"),
        ("Người lái xe [phải] kiểm tra phanh trước mỗi chuyến đi.", "phải rằng"),
        ("Minh [phải] rửa tay trước bữa ăn.", "phải rằng"),
        ("Đội khảo sát [phải] mang áo phao khi xuống thuyền.", "phải rằng"),
        ("Người thợ [phải] ngắt điện trước khi sửa ổ cắm.", "phải rằng"),
        ("Cô Hà [phải] ký vào cả hai bản hợp đồng.", "phải rằng"),
        ("Ban tổ chức [phải] kiểm tra lối thoát hiểm.", "phải rằng"),
        ("Anh Tuấn [phải] sạc pin trước buổi ghi hình.", "phải rằng"),
    ]),
    (3, "copula_before_finite_action", "Là bị chèn sau đã trước động từ hành động không được danh hóa; bỏ hệ từ thừa.", [
        ("Lan [đã] đi bộ đến thư viện.", "đã là"),
        ("Nam [đã] đóng cửa sổ trước cơn mưa.", "đã là"),
        ("Cô giáo [đã] phát bài kiểm tra cho học sinh.", "đã là"),
        ("Người thợ [đã] sơn lại hàng rào.", "đã là"),
        ("Bà Mai [đã] hái rau trong vườn.", "đã là"),
        ("Nhân viên [đã] gửi thư mời cho khách.", "đã là"),
        ("Đầu bếp [đã] rửa sạch các dụng cụ.", "đã là"),
        ("Minh [đã] dọn bàn sau bữa tối.", "đã là"),
        ("Đội cứu hộ [đã] đưa em bé về nhà.", "đã là"),
        ("Chị Hà [đã] cất áo vào tủ.", "đã là"),
    ]),
    (3, "repeated_passive_marker", "Hai được liền nhau cùng dẫn một động từ bị động; chỉ giữ một dấu hiệu bị động.", [
        ("Lan [được] tặng một cuốn từ điển.", "được được"),
        ("Chiếc máy [được] bảo dưỡng vào thứ sáu.", "được được"),
        ("Các em [được] hướng dẫn cách thoát hiểm.", "được được"),
        ("Bức tranh [được] treo ở sảnh chính.", "được được"),
        ("Những thùng hàng [được] chuyển đến kho mới.", "được được"),
        ("Minh [được] mời tham gia câu lạc bộ.", "được được"),
        ("Cây cầu [được] kiểm tra sau trận lũ.", "được được"),
        ("Hồ sơ [được] lưu trong tủ có khóa.", "được được"),
        ("Đoàn khách [được] đón tại nhà ga.", "được được"),
        ("Bộ dụng cụ [được] khử trùng trước ca trực.", "được được"),
    ]),
    (3, "repeated_relative_marker", "Mà bị lặp ở cùng một ranh giới mệnh đề quan hệ; bỏ một hư từ, giữ nguyên chủ-vị trong mệnh đề.", [
        ("Cuốn sách [mà] Lan mượn đã được trả lại.", "mà mà"),
        ("Chiếc xe [mà] Nam sửa đang chạy tốt.", "mà mà"),
        ("Bức ảnh [mà] Minh chụp được treo trong lớp.", "mà mà"),
        ("Căn phòng [mà] cô Hà chọn nằm ở tầng hai.", "mà mà"),
        ("Bài hát [mà] Hạnh trình bày đã được ghi âm.", "mà mà"),
        ("Chiếc áo [mà] bà Mai may có màu xanh.", "mà mà"),
        ("Bản đồ [mà] anh Tuấn mang đã được trải trên bàn.", "mà mà"),
        ("Hạt giống [mà] chị Hoa gieo đã nảy mầm.", "mà mà"),
        ("Tấm vé [mà] Phương mua vẫn còn hạn.", "mà mà"),
        ("Bản báo cáo [mà] nhóm khảo sát viết đã được gửi đi.", "mà mà"),
    ]),
    (4, "comma_splits_simple_subject_predicate", "Dấu phẩy chen giữa chủ ngữ ngắn và vị ngữ, không có thành phần xen hay đồng vị; bỏ dấu phẩy.", [
        ("[Lan] đã mở cửa sổ.", "Lan,"),
        ("[Nam] đang kiểm tra chiếc xe.", "Nam,"),
        ("[Minh] sẽ đến thư viện.", "Minh,"),
        ("[Hạnh] đã xếp sách lên kệ.", "Hạnh,"),
        ("[Tuấn] đang lau bảng.", "Tuấn,"),
        ("[Hoa] sẽ nộp báo cáo vào thứ ba.", "Hoa,"),
        ("[Phương] đã đặt vé tàu.", "Phương,"),
        ("[Dũng] đang sửa hàng rào.", "Dũng,"),
        ("[An] sẽ đọc bài thơ trước lớp.", "An,"),
        ("[Liên] đã gửi bưu kiện.", "Liên,"),
    ]),
    (4, "comma_splits_verb_direct_object", "Dấu phẩy tách động từ ngoại động với bổ ngữ trực tiếp, không có liệt kê hay xen chú; bỏ dấu phẩy.", [
        ("Lan đã [mở] chiếc hộp màu đỏ.", "mở,"),
        ("Nam đang [đọc] bản hướng dẫn sử dụng.", "đọc,"),
        ("Minh sẽ [mang] chiếc bàn vào phòng.", "mang,"),
        ("Chị Hà đã [giặt] tấm rèm cửa.", "giặt,"),
        ("Người thợ đang [sơn] cánh cổng.", "sơn,"),
        ("Cô giáo sẽ [phát] bài tập cho lớp.", "phát,"),
        ("Bà Mai đã [hái] những quả xoài chín.", "hái,"),
        ("Nhân viên đang [kiểm tra] các hóa đơn.", "kiểm tra,"),
        ("Anh Tuấn sẽ [lắp] chiếc đèn mới.", "lắp,"),
        ("Đầu bếp đã [rửa] chiếc nồi lớn.", "rửa,"),
    ]),
    (4, "comma_splits_preposition_complement", "Dấu phẩy tách giới từ khỏi bổ ngữ trực tiếp, không có thành phần xen; bỏ dấu phẩy, không bỏ giới từ.", [
        ("Lan tô màu [bằng] bút sáp.", "bằng,"),
        ("Nam đặt hộp [trên] chiếc bàn gỗ.", "trên,"),
        ("Minh cất vở [trong] ngăn kéo.", "trong,"),
        ("Chị Hà gửi thư [cho] cô Mai.", "cho,"),
        ("Người thợ kéo xe [từ] sân vào kho.", "từ,"),
        ("Bà Mai đứng [dưới] tán cây bàng.", "dưới,"),
        ("Cô giáo trao đổi [với] phụ huynh sau buổi học.", "với,"),
        ("Nhân viên nói [về] lịch trực tuần tới.", "về,"),
        ("Anh Tuấn buộc dây [vào] chiếc cọc gỗ.", "vào,"),
        ("Đoàn khách đi bộ [đến] nhà trưng bày.", "đến,"),
    ]),
    (4, "duplicated_internal_comma", "Hai dấu phẩy liên tiếp tại một ranh giới trạng ngữ-mệnh đề là lỗi dấu câu cơ học; giữ một dấu phẩy.", [
        ("Sáng nay[,] Lan đã nhận được thư.", ",,"),
        ("Chiều qua[,] Nam đã ghé thăm ông bà.", ",,"),
        ("Sau bữa tối[,] Minh rửa bát.", ",,"),
        ("Trước buổi học[,] cô giáo kiểm tra máy chiếu.", ",,"),
        ("Trong giờ nghỉ[,] các em chơi cờ.", ",,"),
        ("Đến cuối tuần[,] nhóm khảo sát sẽ trở về.", ",,"),
        ("Ở sân sau[,] bà Mai trồng một luống rau.", ",,"),
        ("Sau cơn mưa[,] người thợ kiểm tra mái nhà.", ",,"),
        ("Vào thứ hai[,] thư viện mở cửa lúc 8 giờ.", ",,"),
        ("Trước khi đi ngủ[,] chị Hà tắt đèn hành lang.", ",,"),
    ]),
    (4, "unclosed_quotation_mark", "Tên tác phẩm mở bằng ngoặc kép nhưng thiếu ngoặc kép đóng; khôi phục cặp dấu, không đổi chữ.", [
        ("Lan đã đọc truyện “Dòng sông nhỏ[”] trong kỳ nghỉ.", ""),
        ("Nam chọn bài thơ “Mùa lá mới[”] để đọc trước lớp.", ""),
        ("Cô giáo giới thiệu cuốn sách “Khu vườn xanh[”] vào sáng nay.", ""),
        ("Minh nghe bài hát “Con đường đến trường[”] trên đài.", ""),
        ("Đoàn khách xem bộ phim “Ngọn hải đăng[”] tại hội trường.", ""),
        ("Hạnh vẽ bức tranh “Buổi sáng trên hồ[”] cho triển lãm.", ""),
        ("Chị Hà tìm cuốn sách “Căn bếp nhỏ[”] trên kệ.", ""),
        ("Anh Tuấn đặt tên tập ảnh là “Chuyến tàu mùa hạ[”].", ""),
        ("Thư viện đã nhận cuốn truyện “Người giữ rừng[”] vào tuần trước.", ""),
        ("Phương viết bài giới thiệu về vở kịch “Chiếc cầu mới[”].", ""),
    ]),
    (4, "unclosed_parenthesis", "Thành phần chú thích mở bằng ngoặc đơn nhưng thiếu ngoặc đóng; khôi phục cặp dấu.", [
        ("Lan mang hai chiếc áo (một xanh và một trắng[)] đến buổi chụp ảnh.", ""),
        ("Nam đã gọi cho chú Bình (người phụ trách kho[)] vào sáng nay.", ""),
        ("Cô giáo phát giấy khổ lớn (loại A3[)] cho nhóm vẽ.", ""),
        ("Minh chọn chuyến tàu sớm (lúc 6 giờ[)] để đến hội nghị.", ""),
        ("Chị Hà mua một tấm vải (dài 2 mét[)] ở chợ.", ""),
        ("Đội khảo sát đến xã An Bình (huyện Minh Sơn[)] vào thứ ba.", ""),
        ("Người thợ thay bóng đèn (loại 9 W[)] trong phòng đọc.", ""),
        ("Thư viện chuyển sách lên tầng trên (tầng 2[)] vào cuối tuần.", ""),
        ("Ban tổ chức chuẩn bị nước uống (50 chai[)] cho đoàn.", ""),
        ("Anh Tuấn gửi bản vẽ (phiên bản mới[)] cho người quản lý.", ""),
    ]),
    (4, "duplicated_list_colon", "Hai dấu hai chấm liền nhau trước danh sách được giới thiệu rõ ràng là lỗi cơ học; giữ một dấu.", [
        ("Lan mang ba món đồ[:] bút, vở và thước.", "::"),
        ("Nam kiểm tra hai bộ phận[:] phanh và đèn.", "::"),
        ("Cô giáo chia lớp thành hai nhóm[:] nhóm đọc và nhóm viết.", "::"),
        ("Bà Mai mua ba loại rau[:] cải, mồng tơi và rau muống.", "::"),
        ("Đầu bếp chuẩn bị hai nguyên liệu chính[:] gạo và đậu xanh.", "::"),
        ("Chị Hà gửi hai giấy tờ[:] đơn đăng ký và bản sao giấy phép.", "::"),
        ("Nhóm khảo sát đo ba đại lượng[:] chiều dài, chiều rộng và độ sâu.", "::"),
        ("Ban tổ chức mở hai quầy[:] quầy vé và quầy nước.", "::"),
        ("Minh chọn ba màu[:] xanh, trắng và vàng.", "::"),
        ("Người thợ mang hai dụng cụ[:] búa và kìm.", "::"),
    ]),
    (4, "mismatched_quotation_closer", "Tên tác phẩm mở bằng ngoặc kép nhưng kết thúc bằng ngoặc đơn; thay đúng dấu đóng tương ứng.", [
        ("Lan đọc bài thơ “Tiếng chim ban mai[”] cho em nghe.", ")"),
        ("Nam đã mượn cuốn sách “Những vì sao xa[”] ở thư viện.", ")"),
        ("Minh giới thiệu bộ phim “Bến cảng mùa xuân[”] trong buổi sinh hoạt.", ")"),
        ("Cô giáo chọn truyện “Chiếc hộp bí mật[”] cho giờ đọc.", ")"),
        ("Hạnh biểu diễn bài hát “Ngày mới trên cao nguyên[”] ở hội trường.", ")"),
        ("Chị Hà viết nhận xét về vở kịch “Ngôi nhà bên sông[”].", ")"),
        ("Anh Tuấn đặt tên bức ảnh là “Con thuyền lúc bình minh[”].", ")"),
        ("Phương vẽ bức tranh “Lối nhỏ trong vườn[”] vào chiều qua.", ")"),
        ("Bà Mai tìm cuốn sách “Mùa gieo hạt[”] trên kệ.", ")"),
        ("Đoàn khách xem phim “Chuyến đi cuối tuần[”] sau bữa tối.", ")"),
    ]),
]

# Legal fronts are deliberately retained, not converted into deletion errors.
CLEAN_FRONTS = [
    ("voi", "Với bộ dụng cụ mới, người thợ sửa chiếc ghế trong một giờ."),
    ("voi", "Với sự đồng ý của phụ huynh, nhà trường tổ chức chuyến tham quan."),
    ("voi", "Với 20 chỗ ngồi, phòng đọc đủ sức phục vụ cả nhóm."),
    ("voi", "Với kinh nghiệm nhiều năm, cô Hà hướng dẫn nhóm trồng cây."),
    ("voi", "Với chiếc bản đồ trong tay, Nam tìm được đường đến bến tàu."),
    ("voi", "Với nguồn nước sạch, bà Mai trồng rau quanh năm."),
    ("nho", "Nhờ chiếc đèn pin, đội cứu hộ nhìn rõ lối đi trong hang."),
    ("nho", "Nhờ sự hỗ trợ của thư viện, Lan tìm đủ tài liệu cho bài viết."),
    ("nho", "Nhờ tập luyện đều đặn, Minh đã hoàn thành đường chạy 5 km."),
    ("nho", "Nhờ hệ thống tưới mới, vườn hoa không bị khô vào mùa hè."),
    ("nho", "Nhờ lời nhắc của cô giáo, Hạnh mang đủ đồ dùng học tập."),
    ("nho", "Nhờ việc kiểm tra phanh trước chuyến đi, tài xế phát hiện kịp thời chỗ hỏng."),
    ("qua", "Qua cửa sổ, Lan nhìn thấy những chiếc thuyền trên sông."),
    ("qua", "Qua cuộc trao đổi với bác sĩ, bà Mai hiểu rõ cách dùng thuốc."),
    ("qua", "Qua ba buổi thực hành, các em đã biết lắp mạch điện đơn giản."),
    ("qua", "Qua những bức ảnh, đoàn khách hình dung được cảnh vật trên đảo."),
    ("qua", "Qua lớp kính bảo vệ, Minh quan sát mẫu đá trong tủ trưng bày."),
    ("qua", "Qua cuộc họp sáng nay, ban tổ chức đã thống nhất lịch thi đấu."),
    ("neu", "Nếu trời nắng, cả lớp sẽ học thể dục ngoài sân."),
    ("neu", "Nếu nhận đủ hồ sơ, nhân viên sẽ gửi giấy hẹn cho khách."),
    ("neu", "Nếu pin còn đầy, Tuấn sẽ ghi hình buổi biểu diễn."),
    ("neu", "Nếu nước sôi, đầu bếp sẽ cho mì vào nồi."),
    ("neu", "Nếu đường thông thoáng, đoàn xe sẽ đến trước 9 giờ."),
    ("neu", "Nếu tìm thấy chìa khóa, Lan sẽ báo ngay cho người quản lý."),
    ("mac_du", "Mặc dù trời rét, các em vẫn đến lớp đúng giờ."),
    ("mac_du", "Mặc dù căn phòng nhỏ, người thủ thư vẫn bố trí đủ chỗ ngồi."),
    ("mac_du", "Mặc dù lịch làm việc dày, bác sĩ vẫn tham gia buổi tập huấn."),
    ("mac_du", "Mặc dù cơn mưa kéo dài, đội cứu hộ vẫn tiếp tục tìm kiếm."),
    ("mac_du", "Mặc dù chiếc máy đã cũ, nó vẫn hoạt động ổn định."),
    ("mac_du", "Mặc dù mới học đàn, Hạnh đã chơi được bài hát ngắn."),
    ("bang", "Bằng chiếc thuyền nhỏ, ngư dân đưa khách qua sông."),
    ("bang", "Bằng các số liệu đã ghi, nhóm khảo sát tính được diện tích khu đất."),
    ("bang", "Bằng khăn ẩm, nhân viên lau sạch mặt bàn trước buổi họp."),
    ("bang", "Bằng những ví dụ cụ thể, cô giáo giải thích cách tính chu vi."),
    ("bang", "Bằng chiếc cân điện tử, chị Hà kiểm tra khối lượng bưu kiện."),
    ("bang", "Bằng lời nói nhẹ nhàng, bác sĩ trấn an em bé trước khi khám."),
    ("sau_khi", "Sau khi kiểm tra cửa sổ, Nam tắt đèn và rời phòng."),
    ("sau_khi", "Sau khi chuyến tàu đến, đoàn khách tập trung ở cổng nhà ga."),
    ("sau_khi", "Sau khi rửa tay, Lan giúp mẹ chuẩn bị bữa tối."),
    ("sau_khi", "Sau khi mưa ngừng, người thợ leo lên mái nhà để kiểm tra."),
    ("truoc_khi", "Trước khi bắt đầu thí nghiệm, cô giáo nhắc lại quy tắc an toàn."),
    ("truoc_khi", "Trước khi ra khơi, thuyền trưởng kiểm tra áo phao trên tàu."),
    ("truoc_khi", "Trước khi nộp báo cáo, Minh đối chiếu các con số trong bảng."),
    ("truoc_khi", "Trước khi cửa hàng mở cửa, nhân viên xếp hàng lên kệ."),
    ("doi_voi", "Đối với học sinh mới, nhà trường tổ chức một buổi hướng dẫn riêng."),
    ("doi_voi", "Đối với loại cây này, người làm vườn dùng đất có khả năng thoát nước tốt."),
    ("trong", "Trong cuộc họp với tổ dân phố, chị Hoa đề xuất thêm thùng rác."),
    ("o", "Ở phía sau nhà, bà Mai đặt một bồn chứa nước mưa."),
]


def make_row(label: int, family: str, rule: str, rationale: str,
             source: str, target: str, *, clean: bool = False) -> dict:
    return {
        "ID": f"quality:{label}:{family}",
        "Input": source,
        "Output": {"Error_Type_Id": label, "Fixed Sentence": target},
        "Provenance": {
            "parent_ids": [],
            "template_rule": f"clean_{rule}" if clean else rule,
            "rule": f"clean_{rule}" if clean else rule,
            "template_id": f"class{label}_{rule}",
            "family_id": family,
            "origin": "locally_authored_synthetic",
            "authoring_method": "explicit_curated_minimal_contrast",
            "rationale": (
                "Câu đối chứng giữ nguyên câu đích, không gây lỗi; " + rationale
                if clean else rationale
            ),
            "review_status": REVIEW_STATUS,
            "review_limit": "Assistant linguistic review only; not human adjudication; no model-utility claim. Missing-predicate repairs use the authored counterpart, not uniquely inferable original gold.",
            "uses_source_corpus": False,
            "uses_external_teacher": False,
            "training_only": True,
        },
    }


def author_pool() -> list[dict]:
    rows = []
    for label, rule, rationale, entries in GROUPS:
        for ordinal, (marked, replacement) in enumerate(entries, 1):
            if marked.count("[") != 1 or marked.count("]") != 1:
                raise ValueError(f"{rule}:{ordinal} must mark exactly one span")
            before, span = marked.split("[", 1)
            original, after = span.split("]", 1)
            if not original or original == replacement:
                raise ValueError(f"{rule}:{ordinal} must change a nonempty marked span")
            target = before + original + after
            source = before + replacement + after
            if not replacement and before.endswith(" ") and after.startswith(" "):
                source = before + after[1:]
            family = f"{rule}-{ordinal:02d}"
            rows.append(make_row(label, family, rule, rationale, source, target))
            rows.append(make_row(0, family, rule, rationale, target, target, clean=True))
    front_counts = Counter()
    for construction, sentence in CLEAN_FRONTS:
        front_counts[construction] += 1
        family = f"legal_front_{construction}-{front_counts[construction]:02d}"
        rows.append(make_row(
            0, family, f"legal_front_{construction}",
            "Trạng ngữ hoặc mệnh đề phụ đứng đầu câu có mệnh đề chính đầy đủ; không xóa giới từ hay liên từ hợp lệ.",
            sentence, sentence, clean=True,
        ))
    validate_pool(rows)
    return rows


def validate_pool(rows: list[dict]) -> None:
    ids = set()
    sources = {}
    counts = Counter()
    rules = defaultdict(set)
    families = defaultdict(list)
    for row in rows:
        identity = row["ID"]
        label = row["Output"]["Error_Type_Id"]
        source, target = row["Input"], row["Output"]["Fixed Sentence"]
        metadata = row["Provenance"]
        if identity in ids or not re.fullmatch(r"quality:[0-4]:[a-z0-9_-]+", identity):
            raise ValueError(f"Duplicate or malformed ID: {identity}")
        ids.add(identity)
        for text in (source, target):
            if not text or text != text.strip() or "\n" in text:
                raise ValueError(f"Empty, multiline, or padded sentence: {identity}")
            if "..." in text or "…" in text or "[" in text or "]" in text:
                raise ValueError(f"Ellipsis/authoring marker in sentence: {identity}")
            if unicodedata.normalize("NFC", text) != text:
                raise ValueError(f"Non-NFC Vietnamese sentence: {identity}")
        if re.search(r"\([^)]*đúng[^)]*\)", target, flags=re.IGNORECASE):
            raise ValueError(f"Annotation-like target: {identity}")
        if (source == target) != (label == 0):
            raise ValueError(f"Class-0 identity/nonzero contrast violation: {identity}")
        normalized_source = " ".join(source.casefold().split())
        if label != 0 and normalized_source == " ".join(target.casefold().split()):
            raise ValueError(f"Normalized identity for nonzero class: {identity}")
        previous = sources.get(normalized_source)
        value = (label, target)
        if previous is not None and previous != value:
            raise ValueError(f"Conflicting source labels/targets: {identity}")
        sources[normalized_source] = value
        # Punctuation edits preserve every lexical token. Other edits can
        # remove/reorder constituents or change function words, but never digits.
        if re.findall(r"\d+", source) != re.findall(r"\d+", target):
            raise ValueError(f"Number changed by corruption: {identity}")
        if label == 4 and re.findall(r"\w+", source) != re.findall(r"\w+", target):
            raise ValueError(f"Punctuation contrast changed lexical content: {identity}")
        if metadata["parent_ids"] or metadata["uses_source_corpus"] or metadata["uses_external_teacher"]:
            raise ValueError(f"Nonlocal authoring provenance: {identity}")
        if metadata["review_status"] != REVIEW_STATUS or not metadata["rationale"]:
            raise ValueError(f"Incomplete review provenance: {identity}")
        counts[label] += 1
        rules[label].add(metadata["rule"])
        families[metadata["family_id"]].append(row)
    for label in range(1, 5):
        if counts[label] < 80 or len(rules[label]) < 6:
            raise ValueError(f"Insufficient coverage for class {label}")
    if len(CLEAN_FRONTS) < 40 or len(set(sentence for _, sentence in CLEAN_FRONTS)) != len(CLEAN_FRONTS):
        raise ValueError("At least 40 distinct legal clean fronts are required")
    for family, variants in families.items():
        errors = [row for row in variants if row["Output"]["Error_Type_Id"] != 0]
        cleans = [row for row in variants if row["Output"]["Error_Type_Id"] == 0]
        if len(cleans) != 1 or len(errors) > 1:
            raise ValueError(f"A family needs exactly one clean and at most one error: {family}")
        if errors and errors[0]["Output"]["Fixed Sentence"] != cleans[0]["Input"]:
            raise ValueError(f"Clean counterpart differs from correction: {family}")


def write_private_pool(output: Path, rows: list[dict]) -> None:
    # Fixed basename prevents an accidental organizer/gold/bundle overwrite.
    if output.name != "synthetic_pool.json":
        raise ValueError("Output must be named synthetic_pool.json")
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise ValueError("Refusing a symlink or non-file output")
    if output.exists():
        existing = output.read_bytes()
        legacy_sha256 = "6244c13074829fc4f8fcf661337e5f235577e8292319a915030af50fb3552004"
        if hashlib.sha256(existing).hexdigest() != legacy_sha256:
            try:
                previous = json.loads(existing)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError("Refusing to overwrite an unrelated existing file") from exc
            if not isinstance(previous, list) or not previous:
                raise ValueError("Refusing to overwrite an unrelated existing file")
            for row in previous:
                if not isinstance(row, dict) or not isinstance(row.get("Provenance"), dict):
                    raise ValueError("Refusing to overwrite an unrelated existing file")
                provenance = row["Provenance"]
                if (not isinstance(row.get("ID"), str)
                        or not row["ID"].startswith("quality:")
                        or provenance.get("origin") != "locally_authored_synthetic"
                        or provenance.get("authoring_method") != "explicit_curated_minimal_contrast"
                        or provenance.get("review_status") != REVIEW_STATUS
                        or provenance.get("uses_source_corpus") is not False
                        or provenance.get("uses_external_teacher") is not False):
                    raise ValueError("Refusing to overwrite an unrelated existing file")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".synthetic_pool-", suffix=".json", dir=output.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = author_pool()
    write_private_pool(args.output, rows)
    counts = Counter(row["Output"]["Error_Type_Id"] for row in rows)
    print(f"Wrote {len(rows)} private rows to {args.output}; class counts: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
