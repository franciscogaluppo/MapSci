from lib.rs import research_space

key = "sjr"
rs = research_space(key)
rs.load(2011)
print(len(rs.scientists))
rs.load(2020)
print(len(rs.scientists))
