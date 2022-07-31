#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_13_util.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../TempHelpFunctions.hpp"

using torch::indexing::Slice;
using torch::indexing::None;


// Intersection over Union (IoU)
torch::Tensor box_iou(torch::Tensor boxes1, torch::Tensor boxes2) {
    //Compute pairwise IoU across two lists of anchor or bounding boxes."""
	auto box_area = [](torch::Tensor box) noexcept {
		return (box.index({Slice(), 2}) - box.index({Slice(), 0})) *
				(box.index({Slice(), 3}) - box.index({Slice(), 1}));
	};
    // Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    // (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    auto areas1 = box_area(boxes1);
    auto areas2 = box_area(boxes2);
    // Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    // boxes1, no. of boxes2, 2)
    auto inter_upperlefts = torch::max(boxes1.index({Slice(), None, Slice(None, 2)}), boxes2.index({Slice(), Slice(None, 2)}));
    auto inter_lowerrights = torch::min(boxes1.index({Slice(), None, Slice(2, None)}), boxes2.index({Slice(), Slice(2, None)}));
    auto inters = (inter_lowerrights - inter_upperlefts).clamp(0);
    // Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    auto inter_areas = inters.index({Slice(), Slice(), 0}) *  inters.index({Slice(), Slice(), 1});
    auto union_areas = areas1.index({Slice(), None}) + areas2 - inter_areas;
    return inter_areas / union_areas;
}

torch::Tensor assign_anchor_to_bbox(torch::Tensor ground_truth, torch::Tensor anchors,
													torch::Device device, float iou_threshold=0.5) {
    //Assign closest ground-truth bounding boxes to anchor boxes.
    int num_anchors = anchors.size(0);
    int num_gt_boxes = ground_truth.size(0);

    // Element x_ij in the i-th row and j-th column is the IoU of the anchor
    // box i and the ground-truth bounding box j
    auto jaccard = box_iou(anchors, ground_truth);
    // Initialize the tensor to hold the assigned ground-truth bounding box for each anchor
    auto anchors_bbox_map = torch::full({num_anchors,}, -1, at::TensorOptions(torch::kLong)).to(device);

    // Assign ground-truth bounding boxes according to the threshold
    torch::Tensor max_ious, indices;
    std::tie(max_ious, indices) = torch::max(jaccard, 1);
    auto anc_i = torch::nonzero(max_ious >= 0.5).reshape(-1);

    auto box_j = indices.masked_select(max_ious >= 0.5);
    anchors_bbox_map.index_put_({anc_i}, box_j);
    auto col_discard = torch::full({num_anchors,}, -1);
    auto row_discard = torch::full({num_gt_boxes,}, -1);

    for( int i= 0; i < num_gt_boxes; i++ ) {
       	auto max_idx = torch::argmax(jaccard);  // Find the largest IoU
        auto box_idx = (max_idx % num_gt_boxes).to(torch::kLong);
        auto anc_idx = (max_idx / num_gt_boxes).to(torch::kLong);

        anchors_bbox_map.index_put_({anc_idx}, box_idx);
        jaccard.index_put_({Slice(), box_idx}, col_discard);
        jaccard.index_put_({anc_idx, Slice()}, row_discard);
    }

    return anchors_bbox_map;
}

torch::Tensor offset_boxes(torch::Tensor anchors, torch::Tensor assigned_bb, float eps=1e-6) {
	//Transform for anchor box offsets.
    torch::Tensor c_anc = box_corner_to_center(anchors);
    torch::Tensor c_assigned_bb = box_corner_to_center(assigned_bb);

    auto offset_xy = 10 * (c_assigned_bb.index({Slice(), Slice(None, 2)}) -
    						c_anc.index({Slice(), Slice(None, 2)})) / c_anc.index({Slice(), Slice(2, None)});

    auto offset_wh = 5 * torch::log(eps +
    		c_assigned_bb.index({Slice(), Slice(2, None)}) / c_anc.index({Slice(), Slice(2, None)}));

    auto offset = torch::cat({offset_xy, offset_wh}, 1);
    return offset;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> multibox_target(torch::Tensor anchors, torch::Tensor labels) {
    //Label anchor boxes using ground-truth bounding boxes.

    int batch_size = labels.size(0);
    anchors = anchors.squeeze(0);

    std::cout << "batch_size: " << batch_size << '\n';

    std::vector<torch::Tensor> batch_offset, batch_mask, batch_class_labels;
    torch::Device device = anchors.device();
	int num_anchors = anchors.size(0);
	std::cout << "num_anchors: " << num_anchors << '\n';

    for(int i = 0; i < batch_size; i++ ) {

        auto label = labels.index({i, Slice(), Slice()});

        auto anchors_bbox_map = assign_anchor_to_bbox( label.index({Slice(), Slice(1, None)}), anchors, device );

        auto bbox_mask = ((anchors_bbox_map >= 0).to(torch::kFloat).unsqueeze(-1)).repeat({1, 4});

        // Initialize class labels and assigned bounding box coordinates with zeros
        auto class_labels = torch::zeros(num_anchors, at::TensorOptions(torch::kLong)).to(device);

        auto assigned_bb = torch::zeros({num_anchors, 4}, at::TensorOptions(torch::kFloat32)).to(device);

        // Label classes of anchor boxes using their assigned ground-truth
        // bounding boxes. If an anchor box is not assigned any, we label its
        // class as background (the value remains zero)
        auto indices_true = torch::nonzero(anchors_bbox_map >= 0).toType(torch::kLong);

        auto bb_idx = anchors_bbox_map.index_select(0, indices_true.squeeze());

        class_labels.index_put_({indices_true.squeeze()}, label.index({bb_idx, 0}).toType(torch::kLong) + 1);

        assigned_bb.index_put_({indices_true.squeeze()}, label.index({bb_idx, Slice(1, None)}));

        // Offset transformation
        auto offset = offset_boxes(anchors, assigned_bb) * bbox_mask;
        batch_offset.push_back(offset.reshape(-1).toType(torch::kFloat32));
        batch_mask.push_back(bbox_mask.reshape(-1).toType(torch::kFloat32));
        batch_class_labels.push_back(class_labels);
    }
    auto bbox_offset = torch::stack(batch_offset);
    auto bbox_mask = torch::stack(batch_mask);
    auto class_labels = torch::stack(batch_class_labels);
    return std::make_tuple(bbox_offset, bbox_mask, class_labels);
}

torch::Tensor offset_inverse(torch::Tensor anchors, torch::Tensor offset_preds) {
    //Predict bounding boxes based on anchor boxes with predicted offsets.
    auto anc = box_corner_to_center(anchors);
    auto pred_bbox_xy = (offset_preds.index({Slice(), Slice(None, 2)}) * anc.index({Slice(), Slice(2, None)}) / 10) +
    					anc.index({Slice(), Slice(None, 2)});
    auto pred_bbox_wh = torch::exp(offset_preds.index({Slice(), Slice(2, None)}) / 5) * anc.index({Slice(), Slice(2, None)});
    auto pred_bbox = torch::cat({pred_bbox_xy, pred_bbox_wh}, 1);
    auto predicted_bbox = box_center_to_corner(pred_bbox);
    return predicted_bbox;
}

torch::Tensor nms(torch::Tensor boxes, torch::Tensor scores, float iou_threshold) {
    //Sort confidence scores of predicted bounding boxes.
    auto B = torch::argsort(scores, -1, true);
    std::vector<long> keep;	// Indices of predicted bounding boxes that will be kept
    keep.clear();
    while( B.numel() > 0 ) {
    	auto i = B[0].data().item<long>();
    	keep.push_back(i);
    	if( B.numel() == 1 ) break;

    	auto boxes1 = boxes.index({i, Slice()}).reshape({-1, 4});
    	auto boxes2 = boxes.index({B.index({Slice(1, None)}), Slice()}).reshape({-1, 4});
    	torch::Tensor iou = box_iou(boxes1, boxes2).reshape(-1);

    	auto inds = torch::nonzero(iou <= iou_threshold).reshape(-1);
    	B = B.index_select(0, inds + 1);
    }
    int64_t lgth = keep.size();
    auto tt = torch::from_blob(keep.data(), {1, lgth}, at::TensorOptions(torch::kLong)).to(boxes.device()).clone();
    return tt.squeeze_();
}


torch::Tensor multibox_detection(torch::Tensor cls_probs, torch::Tensor offset_preds, torch::Tensor anchors,
								float nms_threshold=0.5, float pos_threshold=0.009999999) {
    //Predict bounding boxes using non-maximum suppression.
	torch::Device device = cls_probs.device();
	int batch_size = cls_probs.size(0);
	anchors = anchors.squeeze(0);

	int num_classes  = cls_probs.size(1), num_anchors = cls_probs.size(2);
	std::vector<torch::Tensor> out;

    for(int i = 0; i < batch_size; i++ ) {
    	auto cls_prob = cls_probs[i];
    	auto offset_pred = offset_preds[i].reshape({-1, 4});

    	torch::Tensor conf, class_id;
    	std::tie(conf, class_id) = torch::max(cls_prob.index({Slice(1, None)}), 0);

    	auto predicted_bb = offset_inverse(anchors, offset_pred);

    	auto keep = nms(predicted_bb, conf, nms_threshold);

    	// Find all non-`keep` indices and set the class to background
    	auto all_idx = torch::arange(num_anchors, at::TensorOptions(torch::kLong)).to(device);
    	auto combined = torch::cat({keep, all_idx});

    	torch::Tensor uniques, n_inverse , counts;
    	// sorted = true, return_inverse = false,return_counts = true
    	std::tie(uniques, n_inverse , counts) = at::_unique2(combined, true, false, true);

    	auto non_keep = uniques.masked_select(counts == 1);
    	auto all_id_sorted = torch::cat({keep, non_keep});

    	class_id.index_put_({non_keep}, -1);
    	class_id = class_id.index_select(0, all_id_sorted);
    	conf = conf.index_select(0, all_id_sorted);
    	predicted_bb = predicted_bb.index_select(0, all_id_sorted);

    	// Here `pos_threshold` is a threshold for positive (non-background) predictions
    	auto below_min_idx = (conf < pos_threshold);
    	class_id.masked_fill_(below_min_idx, -1);

    	conf.masked_select(below_min_idx) = 1 - conf.masked_select(below_min_idx);

    	auto pred_info = torch::cat({class_id.unsqueeze(1),
    			                     conf.unsqueeze(1),
    			                     predicted_bb}, 1);

        out.push_back(pred_info);
    }
    return torch::stack(out);
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// -----------------------------------------------------
	// Generating Multiple Anchor Boxes
	// -----------------------------------------------------

	auto rlt = readImg("./data/catdog.jpg");
	cv::Mat img = rlt.first;
	torch::Tensor imgT = rlt.second;
	std::cout << imgT.sizes() << '\n';

	int h = imgT.size(2), w = imgT.size(3);
	std::cout << "h: " << h << ", w: " << w << '\n';

	auto X = torch::rand({1, 3, h, w});		//Construct input data
	auto Y = multibox_prior(X, {0.75, 0.5, 0.25}, {1, 2, 0.5});
	std::cout << Y.sizes() << '\n';

	auto boxes = Y.reshape({h, w, 5, 4});
	std::cout << "boxes.index({250, 250, 0, Slice()}): " << boxes.index({250, 250, 0, Slice()}) << '\n';

	cv::Mat kpImg = img.clone();
	// show all the anchor boxes centered on one pixel in the image
	auto bbox_scale = torch::tensor({w, h, w, h});
	auto bboxes = boxes.index({250, 250, Slice(), Slice()}) * bbox_scale;

	show_bboxes(img, bboxes, {"s=0.75, r=1", "s=0.5, r=1", "s=0.25, r=1", "s=0.75, r=2",
	             "s=0.75, r=0.5"}, {});

	cv::imshow("exp_1", img);
	cv::waitKey(-1);

	// An Example
	auto ground_truth = torch::tensor({{0.0, 0.1, 0.08, 0.52, 0.92},
	                         {1.0, 0.55, 0.2, 0.9, 0.88}});

	auto anchors = torch::tensor({{0.0, 0.1, 0.2, 0.3}, {0.15, 0.2, 0.4, 0.4},
	                    {0.63, 0.05, 0.88, 0.98}, {0.66, 0.45, 0.8, 0.8},
	                    {0.57, 0.3, 0.92, 0.9}});

	img = kpImg.clone();
	bboxes = ground_truth.index({Slice(), Slice(1, None)}) * bbox_scale;

	show_bboxes(img, bboxes, {"dog", "cat"}, {cv::Scalar(0,0,0)});
	cv::imshow("ground_truth", img);
	cv::waitKey(-1);

	show_bboxes(img, anchors * bbox_scale, {"0", "1", "2", "3", "4"}, {});
	cv::imshow("anchors", img);
	cv::waitKey(-1);

	auto labels = multibox_target(anchors.unsqueeze(0), ground_truth.unsqueeze(0));

	std::cout << "labels[2]: " << std::get<2>(labels) << '\n';

	std::cout << "labels[1]: " << std::get<1>(labels) << '\n';

	// -------------------------------------------------------
	// Predicting Bounding Boxes with Non-Maximum Suppression
	// -------------------------------------------------------

	// Now let us [apply the above implementations to a concrete example with four anchor boxes].
	// For simplicity, we assume that the predicted offsets are all zeros. This means that the predicted
	// bounding boxes are anchor boxes. For each class among the background, dog, and cat, we also
	// define its predicted likelihood.

	anchors = torch::tensor({{0.1, 0.08, 0.52, 0.92},
							{0.08, 0.2, 0.56, 0.95},
	                        {0.15, 0.3, 0.62, 0.91},
							{0.55, 0.2, 0.9, 0.88}});
	std::cout << anchors.numel() << '\n';

	auto offset_preds = torch::tensor({0.0, 0.0, 0.0, 0.0,
									   0.0, 0.0, 0.0, 0.0,
									   0.0, 0.0, 0.0, 0.0,
									   0.0, 0.0, 0.0, 0.0});
	auto cls_probs = torch::tensor({{0.0, 0.0, 0.0, 0.0},  		// Predicted background likelihood
	                      	  	  	{0.9, 0.8, 0.7, 0.1},  		// Predicted dog likelihood
									{0.1, 0.2, 0.3, 0.9}}); 	// Predicted cat likelihood

	// plot these predicted bounding boxes with their confidence on the image
	img = kpImg.clone();
	bboxes = anchors * bbox_scale;

	show_bboxes(img, bboxes, {"dog=0.9", "dog=0.8", "dog=0.7", "cat=0.9"}, {});
	cv::imshow("four anchor boxes", img);
	cv::waitKey(-1);

	// output the final predicted bounding box kept by non-maximum suppression
	auto output = multibox_detection(cls_probs.unsqueeze(0), offset_preds.unsqueeze(0), anchors.unsqueeze(0), 0.5);
	std::cout << "output: " << output[0] << '\n';

	img = kpImg.clone();
	output.squeeze_();

	std::vector<std::string> tlabels = {"dog=", "cat="};

	for(int k = 0; k < output.size(0); k++ ) {
		auto t = output[k];
	    if( t[0].item<int>() == -1 )
	        continue;

	    int lidx = t[0].item<float>();
	    auto tbox = t.index({Slice(2, None)}) * bbox_scale;

	    show_bboxes(img, tbox.unsqueeze(0), {tlabels[lidx] + to_string_with_precision(t[1].item<float>(), 2)}, {});
	}

	cv::imshow("The final predicted bounding box", img);
	cv::waitKey(-1);
	cv::destroyAllWindows();

	std::cout << "Done!\n";
	return 0;
}


