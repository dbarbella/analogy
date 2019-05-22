def _nines(digit):
  num = "9" * digit
  return int(num)


def get_path(book_num,out_dir):
  book_num = str(book_num)
  num_len = len(book_num)

  if num_len < 5:
      fst = 0
      fst_path = "0-10000/"
  else:
    fst = int(book_num[0]) * 10000
    fst_path = ("%s-%s") % (fst, str(fst + 10000) + "/")
  out_path = out_dir + fst_path

  if num_len < 4:
    snd = 0
    snd_path = "%s-%s/" % (fst + snd, fst + snd + 999)
  else:
    last_four = (book_num[-4:])
    snd = int((last_four)[0])*(10**3)

    snd_path = "%s-%s/" % (fst + snd, fst + snd + 999)
  out_path += snd_path

  if num_len < 3 or book_num[-3] == "0":
    trd = 0
    trd_path = "%s-%s/" %(fst + snd+ trd,fst + snd +trd + 99)
  else:
    last_three = book_num[-3:]
    trd = int((last_three)[0])*(10**2)
    trd_path = "%s-%s/" %(fst + snd+ trd,fst + snd +trd + 99)

  out_path += trd_path
  return out_path
