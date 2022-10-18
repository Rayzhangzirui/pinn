function y = parentdir(p)
%
x = strsplit(p,'/');
y = join(x(1:end-1),'/');
y = y{1};
end