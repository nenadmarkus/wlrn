--
require 'torch'

require 'io'
require 'struct'

require 'image'

--
function load_bag(path, reqn)
	--
	local file = io.open(path, "rb")
	local magic, dim, n = struct.unpack("c4ii", file:read(12))
	local bytes = file:read("*all")

	file:close()

	--
	if n<8 then -- we require at least 8 keypoints per bag
		--
		return nil
	end

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
		data = torch.Tensor(tbl):float():view(n, dim)

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
	local n = bag.data:size(1)
	local descsize = bag.data:size(2)

	--
	local file = io.open(path, "wb")

	if not file then
		--
		print('cannot write to ' .. path)
		return
	end

	file:write(struct.pack("c4ii", bag.magic, descsize, n))

	--
	for i=1, n do
		--
		local tbl = bag.data[i]:totable()

		--
		file:write(struct.pack(string.rep("f", descsize), unpack(tbl)))
	end

	--
	file:close()
end

--
function load_bags(fname, reqn)
	--
	local p = io.popen('ls ' .. fname .. '/*.bag')

	local bags = {}

	for path in p:lines() do
		print(path)
		table.insert(bags, load_bag(path, reqn))       
	end

	p:close()

	--
	return bags
end