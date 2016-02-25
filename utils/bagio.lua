--
require 'torch'

require 'io'
require 'struct'

require 'image'

-- a brutal hack obtained somewhere on stackoverflow
function is_dir(path)
	--
	local f = io.open(path, "r")
	local ok, err, code = f:read(1)
	f:close()
	return code == 21
end

--
function load_bag(path, reqn)
	--
	local file = io.open(path, "rb")
	local magic, dim, n = struct.unpack("c4ii", file:read(12))
	local bytes = file:read("*all")

	file:close()

	--
	if reqn then
		if n < reqn then
			return nil
		else
			n = reqn
		end
	end

	if string.sub(magic, 1, 1)=='f' then
		--
		local tbl = {}

		for i=1, n do
			--
			for j=1, dim do
				--
				tbl[#tbl + 1] = struct.unpack("f", bytes, 4*(i-1)*dim + 4*(j-1)+1)
			end
		--
		end

		--
		data = torch.Tensor(tbl):view(n, dim)

		--
		--for i=1, n do
		--	--
		--	image.save("work/" .. i .. ".png", data[i]:view(32, 32))
		--end
	end

	-- get label
	local tmp = {}

	for t in string.gmatch(path, "[^/]+") do
		table.insert(tmp, t)
	end

	local label = {}

	for t in string.gmatch(tmp[#tmp], "[^.]+") do
		table.insert(label, t)
	end

	filename = tmp[#tmp]
	label = label[1]

	--
	local bag = {}

	bag.magic = magic
	bag.label = label
	bag.filename = filename
	bag.data = data

	return bag
end

--
function store_bag(bag, path)
	--
	n = bag.data:size(1)
	descsize = bag.data:size(2)

	--
	file = io.open(path, "wb")
	file:write(struct.pack("c4ii", bag.magic, descsize, n))

	--
	for i=1, n do
		--
		tmp = bag.data[i]:totable()

		--
		file:write(struct.pack(string.rep("f", descsize), unpack(tmp)))
	end

	--
	file:close()
end

--
function load_bags(fname, reqn)
	--
	-- see first if 'fname' points to a file
	if not is_dir(fname) then
		--
		return torch.load(fname)
	end

	--
	local p = io.popen('ls ' .. fname .. '/*.bag')

	local bags = {}

	for path in p:lines() do
		--print(path)
		table.insert(bags, load_bag(path, reqn))       
	end

	p:close()

	--
	return bags
end

--
--
--

--
if arg[1] and arg[2] then
	--
	torch.setdefaulttensortype('torch.FloatTensor')

	--
	bags = load_bags(arg[1])

	--
	torch.save(arg[2], bags)
end