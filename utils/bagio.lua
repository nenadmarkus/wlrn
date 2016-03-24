--
require 'torch'

require 'io'
require 'struct'

require 'image'

--
-- Load/store bags of keypoints
--

function load_bag(path)
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

	--
	typ = string.sub(magic, 1, 1)

	if typ~='f' and typ~='B' then
		return nil
	end

	--
	local tbl = {}

	for i=1, n do
		--
		for j=1, dim do
			--
			if typ=='f' then
				--
				tbl[#tbl + 1] = struct.unpack(typ, bytes, 4*(i-1)*dim + 4*(j-1)+1)
			else
				--
				tbl[#tbl + 1] = struct.unpack(typ, bytes, (i-1)*dim + (j-1)+1)
			end
		end
	--
	end

	--
	local data = torch.Tensor(tbl)

	if typ=='f' then
		data = data:float():view(n, dim)
	else
		data = data:byte():view(n, dim)
	end

	--
	--os.execute('mkdir -p patches/')
	--for i=1, n do
	--	--
	--	image.save("patches/p" .. i .. ".png", data[i]:view(32, 32))
	--end

	-- get label
	local tmp = {}

	for t in string.gmatch(path, "[^/]+") do
		table.insert(tmp, t)
	end

	local label = {}

	for t in string.gmatch(tmp[#tmp], "[^.]+") do
		table.insert(label, t)
	end

	local filename = tmp[#tmp]
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
function load_bags(folder, prob)
	--
	if not prob then
		prob = 1.0
	end

	--
	local p = io.popen('ls ' .. folder .. '/*.bag')

	local bags = {}

	for path in p:lines() do
		if math.random()<=prob then
			--print(path)
			table.insert(bags, load_bag(path))
		end
	end

	p:close()

	--
	return bags
end