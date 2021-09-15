import argparse
import pyspark
import json
import itertools


def train():
    review_rdd = sc.textFile(args.train_file)
    review_rdd = review_rdd.map(lambda line: json.loads(line)) \
        .map(lambda x: (x['user_id'], x['business_id'], float(x['stars'])))

    # print(review_rdd.map(lambda x: x[1]).distinct().count())
    # 10253*10252 / 2

    def indexing(input_list):
        idx = {i: it for i, it in enumerate(input_list)}
        idx_inverse = {it: i for i, it in enumerate(input_list)}
        return idx, idx_inverse

    user = review_rdd.map(lambda x: x[0]).distinct().collect()
    business = review_rdd.map(lambda x: x[1]).distinct().collect()
    user_idx, user_idx_inverse = indexing(user)
    business_idx, business_idx_inverse = indexing(business)

    review_rdd = review_rdd.map(lambda x: (user_idx_inverse[x[0]], business_idx_inverse[x[1]], x[2]))  # (1,3,3,5)
    review_dict = review_rdd.map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()  # return a dictionary
    # (i1, i2):[u1, u2, u3]
    # RDD: (user, list(items))
    user_item_rdd = review_rdd.map(lambda x: (x[0], x[1])).groupByKey().mapValues(lambda x: sorted(list(x)))

    # RDD [user, list[items]], where list(items) > 1
    user_item_rdd = user_item_rdd.filter(lambda x: len(x[1]) > 1)

    # u1: [i1, i2, i3] => (i1, i2), (i1, i3), (i2, i3)
    # RDD [user, list ((i1, i2))]
    user_item_pair_rdd = user_item_rdd.map(lambda x: (x[0], list(itertools.combinations(x[1], 2))))

    # [((i1, i2), u1), ((i1, i3), u1)]
    item_pair_user_rdd = user_item_pair_rdd.flatMap(lambda x: [(p, x[0]) for p in x[1]]).groupByKey() \
        .filter(lambda x: len(x[1]) > 3)  # (i1, i2), (u1, u2, u3) len(user_list) > 3

    # (i1, i2), (u1, u2, u3)# , avg_rating1, avg_rating2
    def pearson_corr(rating_list1, rating_list2):
        avg_rating1 = float(sum(rating_list1)) / float(len(rating_list1))
        avg_rating2 = float(sum(rating_list2)) / float(len(rating_list2))
        var_star1 = [x - avg_rating1 for x in rating_list1]
        var_star2 = [x - avg_rating2 for x in rating_list2]

        s, w1, w2 = 0., 0., 0.
        for i in range(len(var_star1)):
            s += var_star1[i] * var_star2[i]
            w1 += var_star1[i] * var_star1[i]
            w2 += var_star2[i] * var_star2[i]

        if w1 == 0. or w2 == 0.:
            return 0.
        else:
            return s / w1 ** 0.5 / w2 ** 0.5

    def compute_weight(i1, i2, users):
        rating_list1 = [review_dict[(u, i1)] for u in users]
        rating_list2 = [review_dict[(u, i2)] for u in users]
        return pearson_corr(rating_list1, rating_list2)

    result = item_pair_user_rdd.map(lambda x: {'b1': x[0][0], 'b2': x[0][1],
                                               'sim': compute_weight(x[0][0], x[0][1], x[1])})

    file = open(args.model_file, 'w')
    file.write(json.dumps(result.collect()))


if __name__ == '__main__':
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw3') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '8g') \
        .set('spark.executor.memory', '4g')

    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='rs')
    parser.add_argument('--train_file', type=str, default='./data/train_review.json',
                        help='the train file ')
    parser.add_argument('--model_file', type=str, default='./data/model.json',
                        help='the output file contains your answers')
    args = parser.parse_args()

    train()
