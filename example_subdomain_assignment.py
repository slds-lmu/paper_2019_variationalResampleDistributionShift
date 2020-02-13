from main import embed_cluster
embed_cluster(['--z_dim', '10', '--epoch', '1', '--dataset', 'fashion-mnist', '--cluster', 'False'])
embed_cluster(['--z_dim', '10', '--epoch', '1', '--dataset', 'fashion-mnist', '--cluster', 'True', '--labeled', 'True'])
