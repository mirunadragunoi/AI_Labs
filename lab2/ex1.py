"""
# import model
from sklearn.naive_bayes import MultinomialNB

# definire model
naive_bayes_model = MultinomialNB

# antrenare model
naive_bayes_model.fit(training_data, training_labels)

# prezicere etichete
naive_bayes_model.predict(testing_data)

# calcul acuratete
naive_bayes_model.score(testing_data, testing_labels)
"""

# ex 1 - am multimea de antrenare ->> inaltimea in cm a unei persoane + eticheta
# impart valorile continue (inaltimea) in 4 intervale -->> prob ca o persoana de 178 cm sa fie fata sau sa fie baiat

# [(160, F), (165, F), (155, F), (172, F), (175, B), (180, B), (177, B), (190, B)]
# intervalele (150-160) (161-170) (171-180) (181-190)

# p(f) = 4/8 = 1/2
# p(b) = 4/8 = 1/2

p_f = 4/8
p_b = 4/8

# 178 in intervalul (171-180) -->> 1f, 3b
# p(interval(171-180) conditionat de fata) = 1/4
# p(interval(171-180) conditionat de baiat) = 3/4

p_interval_f = 1/4
p_interval_b = 3/4

# BAYES
# p (fata conditionat de 178 cm) = p(f) * p(interval(171-180) conditionat de fata)
# p (baiat conditionat de 178 cm) = p(b) * p(interval(171-180) conditionat de baiat)

p_fata_178 = p_f * p_interval_f
p_baiat_178 = p_b * p_interval_b

print("Probabilitatea sa am fata cu inaltimea de 178 cm: %s" % p_fata_178)
print("Probabilitatea sa am baiat cu inaltimea de 178 cm: %s" % p_baiat_178)

