if __name__ == '__main__':
    import model_optimization
    from tools import find_gpu_with_min_usage
    find_gpu_with_min_usage()
    model_before_tune="dec_results/20241227_122843_0.h5"
    model_after_tune="dec_results/20241227_123959_tuned_1.h5"

    model_before_tune = model_optimization.Model(model_before_tune)
    before_tune_misclassified=model_before_tune.get_idx_of_misclassified_labels_by_class(8)
    model_after_tune = model_optimization.Model(model_after_tune)

    # for i in model_before_tune.label_for_each_class_train[8]:
    #     model_before_tune.show_img(i,"before tune","train")
    #     model_after_tune.show_img(i, "after tune","train")
    model_before_tune.show_img(0, "before tune", "train")
    model_after_tune.show_img(0, "after tune", "train")
    [print(f"misclassified index: {i} results: {model_before_tune.predict_y_test_res[i]}") for i in sorted(before_tune_misclassified)]
    print()
    #class 0 2 3 7