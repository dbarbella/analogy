import java.io.File;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import javax.swing.JOptionPane;

public class PDFOrgnizer {
    public static void main(String[] args) {
        String path = System.getProperty("user.dir");
        File file = new File(path);
        String[] content = file.list();
        String tmp = "";
        int index = 0;
        String ex_user = JOptionPane.showInputDialog("Enter Your Custom Keyword");
        String name_user = JOptionPane.showInputDialog("Name The Folder For Custom Keyword");
        String[] Keywords = new String[]{"Bank", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2011", "2012", "2013", "2014", "2015", "2016", "Books", "Assignment", "paper"};
        String[] Folder_name = new String[] {"Bank", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2011", "2012", "2013", "2014", "2015", "2016", "Books", "Assignment", "PaperWork"};
        try {
            if (!ex_user.equals("")) {
                Keywords = new String[]{ex_user};
                Folder_name = new String[]{name_user};
            }
        }
        catch (NullPointerException ex) {
            JOptionPane.showMessageDialog(null, "You have Cancelled The Operation");
            System.exit(1);
        }
        String[] ex = PDFOrgnizer.unique(PDFOrgnizer.getEx(file.list()));
        int i = 0;
		
        while (i < ex.length) {
            if (Arrays.asList(Keywords).contains(ex[i])) {
                index = Arrays.asList(Keywords).indexOf(ex[i]);
                tmp = PDFOrgnizer.createFolder(Folder_name[index]);
                PDFOrgnizer.process(ex[i], tmp, content);
            }
            ++i;
        }
        JOptionPane.showMessageDialog(null, "Files are now orginized based on extentions");
    }

    public static void process(String ex, String d, String[] content) {
        File tmp = new File("");
        int i = 0;
        while (i < content.length) {
          if (!tmp.isDirectory() && content[i].contains(ex) && content[i].endsWith("pdf")){
                tmp = new File(content[i]);
                PDFOrgnizer.move(tmp.getAbsolutePath(), String.valueOf(PDFOrgnizer.genrate(tmp.getAbsolutePath())) + d + "\\" + content[i]);
            }
            ++i;
        }
    }

    public static void move(String from, String to) {
        Path From = Paths.get(from, new String[0]);
        Path To = Paths.get(to, new String[0]);
        try {
            Files.move(From, To, StandardCopyOption.ATOMIC_MOVE);
        }
        catch (Exception ex) {
            System.err.println(ex.getMessage());
        }
    }

    public static String genrate(String path) {
        String[] x = path.split("\\\\");
        String result = "";
        int i = 0;
        while (i < x.length - 1) {
            result = String.valueOf(result) + x[i] + "\\";
            ++i;
        }
        return result;
    }

    public static String[] getEx(String[] a) {
        String tmp = "";
        String tmp2 = "";
        String[] ex = new String[a.length];
        int i = 0;
        while (i < a.length) {
            int j = a[i].length() - 1;
            while (j >= 0) {
                tmp = String.valueOf(tmp) + a[i].charAt(j);
                --j;
            }
            try {
                tmp = tmp.substring(0, tmp.indexOf(46));
            }
            catch (Exception ee) {
                tmp = tmp.substring(0, 3);
            }
            j = tmp.length() - 1;
            while (j >= 0) {
                tmp2 = String.valueOf(tmp2) + tmp.charAt(j);
                --j;
            }
            ex[i] = tmp2;
            tmp2 = "";
            tmp = "";
            ++i;
        }
        return ex;
    }

    public static String[] unique(String[] x) {
        String a = "";
        int i = 0;
        while (i < x.length) {
            if (!a.contains(x[i])) {
                a = String.valueOf(a) + x[i] + " ";
            }
            ++i;
        }
        return a.split(" ");
    }

    public static String createFolder(String name) {
        File d = new File(name);
        d.mkdir();
        return name;
    }
}