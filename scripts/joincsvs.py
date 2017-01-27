from csv import DictReader, DictWriter
from glob import iglob as glob

if __name__ == "__main__":
	entries = []
	fields = "sender_call,reciver_call,sender_lat,sender_lon,reciver_lat,reciver_lon,frequency,db,time"
	for filename in glob('realtime/*.csv'):
		with open(filename) as part:
			entries += list(DictReader(part))
	with open("output.csv", "w") as full:
		full.write(fields + "\n")
		dw = DictWriter(full, fieldnames=fields.split(","))
		dw.writerows(entries)
