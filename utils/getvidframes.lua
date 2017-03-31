--
--

--
require 'os'

--
local function get_frames(vid, prefix, out)
	--
	--
	for i=1, 59 do
		--
		if i<10 then
			--
			os.execute('ffmpeg -ss 00:0'..i..':00.000 -i ' .. vid .. ' -vframes 1 ' .. out .. '/'..prefix..i..'.0.jpg')
			os.execute('ffmpeg -ss 00:0'..i..':00.250 -i ' .. vid .. ' -vframes 1 ' .. out .. '/'..prefix..i..'.1.jpg')
			os.execute('ffmpeg -ss 00:0'..i..':00.500 -i ' .. vid .. ' -vframes 1 ' .. out .. '/'..prefix..i..'.2.jpg')
			os.execute('ffmpeg -ss 00:0'..i..':00.750 -i ' .. vid .. ' -vframes 1 ' .. out .. '/'..prefix..i..'.3.jpg')
			os.execute('ffmpeg -ss 00:0'..i..':01.000 -i ' .. vid .. ' -vframes 1 ' .. out .. '/'..prefix..i..'.4.jpg')
		else
			--
			os.execute('ffmpeg -ss 00:'..i..':00.000 -i ' .. vid .. ' -vframes 1 ' .. out .. '/'..prefix..i..'.0.jpg')
			os.execute('ffmpeg -ss 00:'..i..':00.250 -i ' .. vid .. ' -vframes 1 ' .. out .. '/'..prefix..i..'.1.jpg')
			os.execute('ffmpeg -ss 00:'..i..':00.500 -i ' .. vid .. ' -vframes 1 ' .. out .. '/'..prefix..i..'.2.jpg')
			os.execute('ffmpeg -ss 00:'..i..':00.750 -i ' .. vid .. ' -vframes 1 ' .. out .. '/'..prefix..i..'.3.jpg')
			os.execute('ffmpeg -ss 00:'..i..':01.000 -i ' .. vid .. ' -vframes 1 ' .. out .. '/'..prefix..i..'.4.jpg')
		end
	end
end

--
v = 0
for filename in paths.iterfiles(arg[1]) do
	--
	--
	get_frames(paths.dirname(paths.thisfile()) .. '/' .. arg[1] .. '/' .. filename, 'v'..v..'-', arg[2])
	--
	v = v + 1
end