import pyspark
import json
import argparse


def predict():
    f_rdd = sc.textFile(args.train_file)
    review_rdd = f_rdd.map(lambda line: json.loads(line)) \
        .map(lambda x: (x['user_id'], x['business_id'], float(x['stars'])))

    f_rdd = sc.textFile(args.test_file)
    test_rdd = f_rdd.map(lambda line: json.loads(line)) \
        .map(lambda x: (x['user_id'], x['business_id']))

    f_rdd = sc.textFile(args.model_file)
    model_dict = f_rdd.map(lambda line: json.loads(line)) \
        .map(lambda x: ((x['b1'], x['b2']), float(x['sim']))).collectAsMap()

    user_bus_dict = review_rdd.map(lambda x: (x[0], x[1])).groupByKey().mapValue(list).collectAsMap()  #
    review_dict = review_rdd.map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()

    def compute_weighted_stars(tar_user, tar_bus):
        bus_list = user_bus_dict.get(tar_user)   # none

        if bus_list is None:
            return -1

        weight = []
        for bus in bus_list:
            if model_dict.get((tar_bus, bus)) is None and model_dict.get((bus, tar_bus)) is None:
                continue

            sim = model_dict[(tar_bus, bus)] if model_dict.get((tar_bus, bus)) is not None else model_dict[
                (bus, tar_bus)]
            rating = review_dict[(tar_user, bus)]
            weight.append((sim, rating))

        weight = sorted(weight, reverse=True)

        if len(weight) < 2:
            return -1

        weight = weight[:2]
        weighted_sum, sum_weight = 0., 0.
        for w in weight:
            weighted_sum += w[0] * w[1]
            sum_weight += abs(w[0])

        if sum_weight == 0.:
            return -1

        return weighted_sum / sum_weight

    result_rdd = test_rdd.map(lambda x: (x[0], x[1], compute_weighted_stars(x[0], x[1]))) \
        .map(lambda x: {'user_id': x[0], 'business_id': x[1], 'stars': compute_weighted_stars(x[0], x[1])})
    print(result_rdd.collect())
    file = open(args.output_file, 'w')
    file.write(json.dumps(result_rdd.collect()))

    # evaluation steps
    # def RMSE(l1, l2): sklearn.RMSE
    # compute root mean square error

    # 1. load the prediction from the output_file
    # 2. load the ground truth from the test_review_ratings_file

    # l1, l2 = [], []
    # for each (user, business, star) in ground_truth:
    #    find(user, business, prediction) in the prediction
    #    if true: l1.append(star), l2.append(prediction)
    #    else: continue/ or use average value to fill in


if __name__ == '__main__':
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw3') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '8g') \
        .set('spark.executor.memory', '4g')

    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='RS')
    parser.add_argument('--train_file', type=str, default='./data/train_review.json',
                        help='the input data path')
    parser.add_argument('--test_file', type=str, default='./data/test_review_ratings.json',
                        help=' test path')
    parser.add_argument('--model_file', type=str, default='./data/model.json',
                        help='model path')
    parser.add_argument('--output_file', type=str, default='./data/output.json',
                        help='the output file contains your result ')
    args = parser.parse_args()

    predict()
