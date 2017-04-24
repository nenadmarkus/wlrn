--
require 'image'

--
-- generates triplets from a list of bags with annotated classes
function generate_triplets(bags, n)
	--
	local i, j

	--
	if not n then
		--
		n = #bags
	else
		n = math.min(n, #bags)
	end

	-- select a random subset of bags
	local p = torch.randperm(#bags)
	local subset = {}

	for i=1, n do
		--
		local bag = {}

		bag.magic = bags[ p[i] ].magic
		bag.label = bags[ p[i] ].label
		bag.filename = bags[ p[i] ].filename

		--
		bag.data = bags[ p[i] ].data

		--
		subset[1+#subset] = bag
	end

	--
	local triplets = {}

	for i=1, #subset do
		for j=i+1, #subset do
			if subset[i].label == subset[j].label then
				for k=1, 3 do -- generate 3 triplets for each matching pair
					--
					local stop = false
					local q

					while not stop do
						--
						q = math.random(1, #subset)

						--
						if subset[i].label ~= subset[q].label then
							--
							stop = true
						end
					end

					--
					table.insert(triplets, {subset[i].data, subset[j].data, subset[q].data})
				end
			end
		end
	end

	--
	return triplets
end

--
--
function load_keypoint_bags(folder, nchannels, prob)

	--
	local bags = {}

	for filename in paths.iterfiles(folder) do
		--
		-- discard bag with probability (1-prob)
		if math.random()<=prob and paths.extname(filename)=='jpg' then
			--
			-- extract bag label from filename
			local tmp = {}
			for t in string.gmatch(filename, '[^.]+') do
				table.insert(tmp, t)
			end
			local label = tmp[1]

			--
			-- load patches
			local data = image.load(folder .. '/' .. filename, nchannels, 'byte')

			if 1==nchannels then
				--
				if 3==data:size():size() then
					data = data:view(data:size(2)/data:size(3), data:size(3)*data:size(3))
				else
					data = data:view(data:size(1)/data:size(2), data:size(2)*data:size(2))
				end
			else
				--
				data = data:view(3, data:size(2)/data:size(3), data:size(3)*data:size(3)):transpose(1, 2):contiguous()
				data = data:view(data:size(1), 3*data:size(3))
			end

			--
			--os.execute('mkdir -p patches/')
			--for i=1, data:size(1) do
			--	--
			--	image.save("patches/p" .. i .. ".png", data[i]:view(nchannels, 32, 32))
			--end
			--do return end

			--
			--
			local bag = {}
			bag.data = data
			bag.label = label
			table.insert(bags, bag)
		end
	end

	--
	return bags
end

--
function get_trn_triplets()
	--
	local folder = --TRN-FOLDER--
	local prob = --TRN-PROBABILITY--

	return generate_triplets(load_keypoint_bags(folder, 3, prob))
end

function get_vld_triplets()
	--
	local folder = --VLD-FOLDER--
	local prob = --VLD-PROBABILITY--

	return generate_triplets(load_keypoint_bags(folder, 3, prob))
end

--
return get_trn_triplets, get_vld_triplets