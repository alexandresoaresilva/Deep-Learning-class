syms x1 x2;
X = [x1 x2]
syms w1_11 w1_12;
syms w1_21 w1_22;
W1 = [w1_11 w1_12; w1_21 w1_22]
syms w2_11 w2_12 w2_21 w2_22;
W2 = [w2_11 w2_12; w2_21 w2_22]

Z1 = W1*X'

Z2 = W2*Z1

syms y1 y2 y1_hat y2_hat;

Y = [y1 y2]
Y_hat = [y1_hat y2_hat]

syms h3_prime1 h3_prime2;

H3_prime =[h3_prime1 h3_prime2];

H2_prime =[h3_prime1 h3_prime2];
step1 = [-Y'.*H3_prime']


